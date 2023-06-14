import pywapor
from sentinelsat import SentinelAPI
import os
from datetime import datetime as dt
import tqdm
import shutil
import pywapor
import warnings
from pywapor.general.processing_functions import save_ds, open_ds, create_wkt, unpack, make_example_ds, remove_ds, adjust_timelim_dtype
from pywapor.general.logger import log
from pywapor.enhancers.apply_enhancers import apply_enhancer
import xarray as xr
import rioxarray.merge
import tqdm
import numpy as np
import glob
import rasterio.crs
import types
import logging
import hashlib
import json
from sentinelsat.download import Downloader

def download(folder, latlim, lonlim, timelim, search_kwargs, node_filter = None, to_dl = None):
    """Download data using the SentinelSAT API.

    Parameters
    ----------
    folder : str
        Path to folder in which to download data.
    latlim : list
        Latitude limits of area of interest.
    lonlim : list
        Longitude limits of area of interest.
    timelim : list
        Period for which to prepare data.
    search_kwargs : dict
        Extra search kwargs.
    node_filter : function, optional
        Function to filter files inside a node, by default None.

    Returns
    -------
    list
        Paths to downloaded nodes.
    """

    if not os.path.isdir(folder):
        os.makedirs(folder)

    timelim = adjust_timelim_dtype(timelim)

    def _progress_bar(self, **kwargs):
        if "checksumming" in kwargs.get("desc", "no_desc"):
            kwargs.update({"disable": True, "delay": 15})
        elif "Fetching" in kwargs.get("desc", "no_desc"):
            kwargs.update({"disable": True, "delay": 15})
        elif "Downloading products" in kwargs.get("desc", "no_desc"):
            kwargs.update({"disable": True, "position": 0, "desc": "Downloading scenes", "unit": "scene"})
        else:
            kwargs.update({"disable": False, "position": 0})
        kwargs.update({"leave": False})
        return tqdm.tqdm(**kwargs)

    un, pw = pywapor.collect.accounts.get('SENTINEL')
    api = SentinelAPI(un, pw, 'https://apihub.copernicus.eu/apihub')
    api._tqdm = types.MethodType(_progress_bar, api)
    
    logger = api.logger
    logger.setLevel('INFO')
    handler = logging.FileHandler(filename = os.path.join(folder, "log_sentinelapi.txt"))
    formatter = logging.Formatter(fmt='%(levelname)s %(asctime)s: %(message)s')
    handler.setLevel('INFO')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    footprint = create_wkt(latlim, lonlim)

    def _search_query(api, kwargs):
        search_hash = hashlib.sha1(json.dumps({k:str(v) for k,v in kwargs.items()}, sort_keys=True).encode()).hexdigest()
        fp = os.path.join(folder, search_hash + ".json")
        if os.path.isfile(fp):
            with open(fp) as f:
                products = json.load(f)
        else:
            products = {k: v["title"] for k,v in api.query(**kwargs).items()}
            with open(fp, "w") as f:
                json.dump(products, f)
        return products

    def _check_scene(folder, v, to_dl):
        fns = [os.path.split(x)[-1] for x in glob.glob(os.path.join(folder, v + "*", "**","*"), recursive=True)]
        check = np.all([np.any([y in x for x in fns]) for y in to_dl])
        return check

    products = _search_query(api, {"area": footprint, "date": tuple(timelim), **search_kwargs})
    log.info(f"--> Found {len(products)} {search_kwargs['producttype']} scenes.")
    
    if not isinstance(to_dl, type(None)):
        to_keep = {k: v for k, v in products.items() if not _check_scene(folder, v, to_dl)}
    else:
        to_keep = {k: v for k, v in products.items() if len(glob.glob(os.path.join(folder, v + "*"))) == 0}
    log.info(f"--> {len(products) - len(to_keep)} scenes already downloaded, collecting remaining {len(to_keep)}.")

    dler = Downloader(api, node_filter = node_filter)
    dler._tqdm = types.MethodType(_progress_bar, dler)

    statuses, exceptions, out = dler.download_all(to_keep, folder)

    if len(exceptions) > 0:
        log.info(f"--> An exception occured for {len(exceptions)} scenes, check `log_sentinelapi.txt` for more info.")

    scenes = [glob.glob(os.path.join(folder, v + "*"))[0] for k, v in products.items() if not k in exceptions.keys()]

    log.info(f"--> Finished downloading {len(scenes)} {search_kwargs['producttype']} scenes.")

    return scenes

def process_sentinel(scenes, variables, source_name, time_func, final_fn, post_processors, bb = None):
    """Process downloaded Sentinel scenes into netCDFs.

    Parameters
    ----------
    scenes : list
        Paths to downloaded nodes.
    variables : dict
        Keys are variable names, values are additional settings.
    source_name : "SENTINEL2" | "SENTINEL3"
        Whether the data comes from S2 or S3.
    time_func : function
        Function that parses a np.datetime64 from a filename.
    final_fn : str
        Path to the file in which to store all the combined data.
    post_processors : dict
        Functions to apply when the data has been processed.
    bb : list, optional
        Boundingbox to clip to, [xmin, ymin, xmax, ymax], by default None.

    Returns
    -------
    xr.Dataset
        Ouput data.

    Raises
    ------
    ValueError
        Invalid value for `source_name`.
    """

    chunks = {"time": 1, "x": 1000, "y": 1000}

    example_ds = None
    dss1 = dict()

    log.info(f"--> Processing {len(scenes)} scenes.").add()

    target_crs = rasterio.crs.CRS.from_epsg(4326)

    for i, scene_folder in enumerate(scenes):
        
        folder, fn = os.path.split(scene_folder)

        ext = os.path.splitext(scene_folder)[-1]
        
        fp = os.path.join(folder, os.path.splitext(fn)[0] + ".nc")
        if os.path.isfile(fp):
            ds = open_ds(fp)
            dtime = ds.time.values[0]
            if dtime in dss1.keys():
                dss1[dtime].append(ds)
            else:
                dss1[dtime] = [ds]
            continue

        if ext == ".zip":
            scene_folder = unpack(fn, folder)
            remove_folder = True
        else:
            scene_folder = scene_folder
            remove_folder = False

        if source_name == "SENTINEL2":
            ds = pywapor.collect.product.SENTINEL2.s2_processor(scene_folder, variables)
        elif source_name == "SENTINEL3":
            ds = pywapor.collect.product.SENTINEL3.s3_processor(scene_folder, variables, bb = bb)
        else:
            raise ValueError

        # Apply variable specific functions.
        for vars in variables.values():
            for func in vars[2]:
                ds, label = apply_enhancer(ds, vars[1], func)

        # NOTE: see https://github.com/corteva/rioxarray/issues/545
        # NOTE: see rioxarray issue here: https://github.com/corteva/rioxarray/issues/570
        ds = ds.sortby("y", ascending = False)
        _ = [ds[var].rio.write_nodata(np.nan, inplace = True) for var in ds.data_vars]

        # Clip and pad to bounding-box
        if isinstance(example_ds, type(None)):
            example_ds = make_example_ds(ds, folder, target_crs, bb = bb)
        ds = ds.rio.reproject_match(example_ds).chunk({k: v for k, v in chunks.items() if k in ["x", "y"]})

        # NOTE: see rioxarray issue here: https://github.com/corteva/rioxarray/issues/570
        _ = [ds[var].attrs.pop("_FillValue") for var in ds.data_vars if "_FillValue" in ds[var].attrs.keys()]
        
        ds = ds.assign_coords({"x": example_ds.x, "y": example_ds.y})

        dtime = time_func(fn)
        ds = ds.expand_dims({"time": 1})
        ds = ds.assign_coords({"time":[dtime]})
        ds.attrs = {}

        # Save to netcdf
        ds = save_ds(ds, fp, chunks = chunks, encoding = "initiate", label = f"({i+1}/{len(scenes)}) Processing {fn} to netCDF.")

        if dtime in dss1.keys():
            dss1[dtime].append(ds)
        else:
            dss1[dtime] = [ds]

        if remove_folder:
            shutil.rmtree(scene_folder)

    log.sub()
    
    # Merge spatially.
    dss = [xr.concat(dss0, "stacked").median("stacked") for dss0 in dss1.values()]

    # Merge temporally.
    ds = xr.concat(dss, "time")

    # Define output path.
    fp = os.path.join(folder, final_fn)
    
    # Apply general product functions.
    for var, funcs in post_processors.items():
        for func in funcs:
            ds, label = apply_enhancer(ds, var, func)
            log.info(label)

    # Remove unrequested variables.
    ds = ds[list(post_processors.keys())]
    
    for var in ds.data_vars:
        ds[var].attrs = {}

    ds = ds.rio.write_crs(target_crs)

    ds = ds.sortby("time")

    # Save final netcdf.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
        warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
        ds = save_ds(ds, fp, chunks = chunks, encoding = "initiate", label = f"Merging files.")

    # Remove intermediate files.
    for dss0 in dss1.values():
        for x in dss0:
            remove_ds(x)

    return ds

# if __name__ == "__main__":

    # folder = r"/Users/hmcoerver/On My Mac/sentinel_dl_test/SENTINEL3"
    # latlim = [28.9, 29.7]
    # lonlim = [30.2, 31.2]
    # timelim = ["2022-06-01", "2022-07-11"]
    # product_name = 'SL_2_LST___'
    # node_filter = None

    # search_kwargs = {
    #                     "platformname": "Sentinel-3",
    #                     "producttype": product_name,
    #                     "limit": 5,
    #                     }

    # scenes = download(folder, latlim, lonlim, timelim, search_kwargs, node_filter = node_filter)

    # import numpy as np

    # folder = r"/Users/hmcoerver/On My Mac/sentinel_dl_test/SENTINEL2"
    # latlim = [28.9, 29.7]
    # lonlim = [30.2, 31.2]
    # timelim = ["2022-06-01", "2022-07-11"]
    # product_name = 'S2MSI2A'
    # req_vars = ["ndvi", "r0"]

    # adjust_logger(True, folder, "INFO")

    # variables = pywapor.collect.product.SENTINEL2.default_vars(product_name, req_vars)

    # def node_filter(node_info):
    #     fn = os.path.split(node_info["node_path"])[-1]
    #     to_dl = list(variables.keys())
    #     return np.any([x in fn for x in to_dl])

    # search_kwargs = {
    #                 "platformname": "Sentinel-2",
    #                 "producttype": product_name,
    #                 "limit": 5,
    #                 }

    # scenes = download(folder, latlim, lonlim, timelim, search_kwargs, node_filter = node_filter)

    ##

    # if isinstance(timelim[0], str):
    #     timelim[0] = dt.strptime(timelim[0], "%Y-%m-%d")
    #     timelim[1] = dt.strptime(timelim[1], "%Y-%m-%d")

    # def _progress_bar(self, **kwargs):
    #     if "checksumming" in kwargs.get("desc", "no_desc"):
    #         kwargs.update({"disable": True, "delay": 15})
    #     elif "Fetching" in kwargs.get("desc", "no_desc"):
    #         kwargs.update({"disable": True, "delay": 15})
    #     elif "Downloading products" in kwargs.get("desc", "no_desc"):
    #         kwargs.update({"disable": True, "position": 0, "desc": "Downloading scenes", "unit": "scene"})
    #     else:
    #         kwargs.update({"disable": False, "position": 0})
    #     kwargs.update({"leave": False})
    #     return tqdm.tqdm(**kwargs)

    # sentinel_id = 'a2b9a381-3e9a-4fc8-a5ec-4040d43b83b7'
    # search_kwargs.update({"uuid": sentinel_id})
    
    # un, pw = pywapor.collect.accounts.get('SENTINEL')
    # api = SentinelAPI(un, pw, 'https://apihub.copernicus.eu/apihub')
    # api._tqdm = types.MethodType(_progress_bar, api)

    # footprint = create_wkt(latlim, lonlim)
    # products = api.query(footprint, **search_kwargs)
    # log.info(f"--> Found {len(products)} {search_kwargs['producttype']} scenes.")

    # dler = Downloader(api, node_filter = node_filter)
    # dler._tqdm = types.MethodType(_progress_bar, dler)

    # statuses, exceptions, out = dler.download_all(products, folder)
