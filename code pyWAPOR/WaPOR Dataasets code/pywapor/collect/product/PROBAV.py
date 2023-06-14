import os
from pywapor.collect import accounts
import datetime
import pandas as pd
import pywapor.collect.accounts as accounts
from pywapor.general.logger import log
import re
import warnings
from functools import partial
import requests
from pywapor.collect.protocol.crawler import download_urls, crawl
import pywapor.general.bitmasks as bm
import xarray as xr
from pywapor.enhancers.apply_enhancers import apply_enhancer
import numpy as np
from pywapor.general.processing_functions import process_ds, save_ds, open_ds, remove_ds
import datetime
import rioxarray.merge
import copy

def download(folder, latlim, lonlim, timelim, product_name, req_vars = ["ndvi", "r0"],
                variables = None, post_processors = None, timedelta = None):
    """Download MODIS data and store it in a single netCDF file.

    Parameters
    ----------
    folder : str
        Path to folder in which to store results.
    latlim : list
        Latitude limits of area of interest.
    lonlim : list
        Longitude limits of area of interest.
    timelim : list
        Period for which to prepare data.
    product_name : str
        Name of the product to download.
    req_vars : list, optional
        Which variables to download for the selected product, by default ["ndvi", "r0"].
    variables : dict, optional
        Metadata on which exact layers need to be requested from the server, by default None.
    post_processors : dict, optional
        Functions per variable that should be applied to the variable, by default None.

    Returns
    -------
    xr.Dataset
        Downloaded data.
    """

    folder = os.path.join(folder, "PROBAV")

    fn = os.path.join(folder, f"{product_name}.nc")
    req_vars_orig = copy.deepcopy(req_vars)
    if os.path.isfile(fn):
        existing_ds = open_ds(fn)
        req_vars_new = list(set(req_vars).difference(set(existing_ds.data_vars)))
        if len(req_vars_new) > 0:
            req_vars = req_vars_new
            existing_ds = existing_ds.close()
        else:
            return existing_ds[req_vars_orig]

    dates = pd.date_range(timelim[0], timelim[1], freq="D")

    bb = (lonlim[0], latlim[0], lonlim[1], latlim[1])

    if isinstance(variables, type(None)):
        variables = default_vars(product_name, req_vars)

    if isinstance(post_processors, type(None)):
        post_processors = default_post_processors(product_name, req_vars)
    else:
        default_processors = default_post_processors(product_name, req_vars)
        post_processors = {k: {True: default_processors[k], False: v}[v == "default"] for k,v in post_processors.items() if k in req_vars}

    coords = {"x": ["lon", None], "y": ["lat", None]}

    # TODO use urllib
    base_url = f"https://www.vito-eodata.be/PDF/datapool/Free_Data/PROBA-V_100m/{product_name}"
    coord_req = "?coord=" + ",".join([str(x) for x in bb])
    url = f"{base_url}/{coord_req}"

    session = requests.sessions.Session()
    session.auth = accounts.get("VITO")

    # Scrape urls.
    urls = find_tiles(url, dates, session)

    # Download .HDF5 tiles.
    fps = download_urls(urls, folder, session, parallel = 4)

    # Convert to netcdf.
    dss = dict()
    for fp in fps:
        date = datetime.datetime.strptime(fp.split("_")[-3], "%Y%m%d")
        ds = open_hdf5_groups(fp, variables, coords).expand_dims({"time": [date]})
        if date in dss.keys():
            dss[date].append(ds)
        else:
            dss[date] = [ds]

    # Merge tiles.
    dss0 = list()
    for date, datasets in dss.items():
        bb = (lonlim[0], latlim[0], lonlim[1], latlim[1])
        ds = rioxarray.merge.merge_datasets(datasets)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category = FutureWarning)
            ds = ds.rio.clip_box(*bb) # NOTE using this seperately, because `bounds`-kw for `merge_datasets` bugs.
        dss0.append(ds)

    # Merge dates.
    ds = xr.concat(dss0, dim = "time")

    # Apply product specific functions.
    for var, funcs in post_processors.items():
        for func in funcs:
            ds, _ = apply_enhancer(ds, var, func)

    if product_name == "S5_TOC_100_m_C1":
        ds["time"] = ds["time"] + np.timedelta64(int(2.5 * 24), "h")

    ds = ds[req_vars]

    ds = save_ds(ds, fn, label = f"Merging files.")

    for k, v in dss.items():
        for nc in v:
            remove_ds(nc)

    return ds[req_vars_orig]

def open_hdf5_groups(fp, variables, coords):
    """Convert hdf5 with groups (specified in `variable`) to netCDF without groups.

    Parameters
    ----------
    fp : str
        Path to input file.
    variables : dict
        Metadata on which exact layers need to be requested from the server.
    coords : dict
        Metadata on the coordinates in `fp`.

    Returns
    -------
    xr.Dataset
        Ouput data.
    """

    nc_fp = fp.replace(".HDF5", ".nc")
    if os.path.isfile(nc_fp):
        ds = open_ds(nc_fp, "all")
    else:
        ds = xr.open_dataset(fp, chunks = "auto")

        spatial_ref_name = [k for k, v in variables.items() if v[1] == "spatial_ref"][0]
        if (spatial_ref_name in list(ds.variables)) and (spatial_ref_name not in list(ds.coords)):
            ds = ds.rio.write_grid_mapping(spatial_ref_name)
            ds = ds.set_coords((spatial_ref_name))

        for k in variables.keys():
            if k in ds.variables:
                continue
            with xr.open_dataset(fp, group = k, engine = "netcdf4", chunks = "auto") as ds_grp:
                vrs = list(ds_grp.variables)[0] # NOTE assuming each group has only 1 variable!
                ds[k] = ds_grp.rename({vrs: k})[k]

        ds = process_ds(ds, coords, variables)

        ds = save_ds(ds, nc_fp, label = f"Converting {os.path.split(fp)[-1]} to netcdf.")

    return ds

def mask_bitwise_qa(ds, var, flags):
    """Mask PROBAV data using a qa variable.

    Parameters
    ----------
    ds : xr.Dataset
        Input data.
    var : str
        Variable in `ds` to mask.
    flags : list
        Which flags not to mask.

    Returns
    -------
    xr.Dataset
        Masked dataset.
    """
    flag_bits = bm.PROBAV_qa_translator()
    mask = bm.get_mask(ds["qa"].astype("uint8"), flags, flag_bits)
    ds[var] = ds[var].where(~mask, np.nan)
    return ds

def calc_r0(ds, *args):
    ds["r0"] = 0.429 * ds["blue"] + 0.333 * ds["red"] + 0.133 * ds["nir"] + 0.105 * ds["swir"]
    return ds

def default_post_processors(product_name, req_vars = ["ndvi", "r0"]):
    """Given a `product_name` and a list of requested variables, returns a dictionary with a 
    list of functions per variable that should be applied after having collected the data
    from a server.

    Parameters
    ----------
    product_name : str
        Name of the product.
    req_vars : list, optional
        List of variables to be collected, by default ["ndvi", "r0"].

    Returns
    -------
    dict
        Functions per variable that should be applied to the variable.
    """

    post_processors = {
        "S5_TOC_100_m_C1": {
            "r0": [
                    calc_r0, 
                    partial(mask_bitwise_qa, 
                    flags = ["bad BLUE", "bad RED", "bad NIR", "bad SWIR", 
                            "sea", "undefined", "cloud", "ice/snow", "shadow"]),
            ],
            "ndvi": [
                    partial(mask_bitwise_qa, 
                    flags = ["bad RED", "bad NIR", "sea", "undefined", 
                    "cloud", "ice/snow", "shadow"]),
            ],
        }
    }

    out = {k:v for k,v in post_processors[product_name].items() if k in req_vars}

    return out

def default_vars(product_name, req_vars = ["ndvi", "r0"]):
    """Given a `product_name` and a list of requested variables, returns a dictionary
    with metadata on which exact layers need to be requested from the server, how they should
    be renamed, and how their dimensions are defined.

    Parameters
    ----------
    product_name : str
        Name of the product.
    req_vars : list, optional
        List of variables to be collected, by default ["ndvi", "r0"].

    Returns
    -------
    dict
        Metadata on which exact layers need to be requested from the server.
    """
    
    variables = {
        "S5_TOC_100_m_C1": {
            "LEVEL3/NDVI":              [("lon", "lat"), "ndvi"],
            "LEVEL3/RADIOMETRY/BLUE":   [("lon", "lat"), "blue"],
            "LEVEL3/RADIOMETRY/NIR":    [("lon", "lat"), "nir"],
            "LEVEL3/RADIOMETRY/RED":    [("lon", "lat"), "red"],
            "LEVEL3/RADIOMETRY/SWIR":   [("lon", "lat"), "swir"],
            "LEVEL3/GEOMETRY/VNIR":     [("lon", "lat"), "vnir_vza"],
            "LEVEL3/GEOMETRY/SWIR":     [("lon", "lat"), "swir_vza"],
            "LEVEL3/QUALITY":           [("lon", "lat"), "qa"],
            "crs":                      [(), "spatial_ref"]
        },
    }

    req_dl_vars = {
        "S5_TOC_100_m_C1": {
            "ndvi": ["LEVEL3/NDVI", "LEVEL3/QUALITY", "crs"],
            "r0":   ["LEVEL3/RADIOMETRY/BLUE", "LEVEL3/RADIOMETRY/NIR", 
                    "LEVEL3/RADIOMETRY/RED", "LEVEL3/RADIOMETRY/SWIR",
                    "LEVEL3/QUALITY", "crs"],
        }
    }

    out = {val:variables[product_name][val] for sublist in map(req_dl_vars[product_name].get, req_vars) for val in sublist}
    
    return out

def find_tiles(url, dates, session):
    """Crawl webpage to find tile URLs.

    Parameters
    ----------
    url : str
        Path to start crawl.
    dates : pd.date_range
        Dates for which to return URLs.
    session : requests.Session
        Session to use during crawl.

    Returns
    -------
    list
        List with URLs to scenes.
    """

    log.info("--> Searching PROBAV tiles.")

    regex = "https:.*\/\d{4}\/\?coord="
    filter_regex = "\d{4}(?=\/\?coord=)"
    urls = {"_": url}
    label_filter = dates.strftime("%Y")
    years = crawl(urls, regex, filter_regex, session, label_filter = label_filter)

    regex = "https:.*\/\d{4}\/\d{2}\/\?coord="
    filter_regex = "\d{4}\/\d{2}(?=\/\?coord=)"
    label_filter = dates.strftime("%Y%m")
    months = crawl(years, regex, filter_regex, session, label_filter = label_filter)

    regex = "https:.*\/\d{4}\/\d{2}\/\d{2}\/\?coord="
    filter_regex = "\d{4}\/\d{2}\/\d{2}(?=\/\?coord=)"
    label_filter = dates.strftime("%Y%m%d")
    days = crawl(months, regex, filter_regex, session, label_filter = label_filter)

    regex = "https:.*\/\d{4}\/\d{2}\/\d{2}\/.*\/\?coord="
    filter_regex = "\d{4}\/\d{2}\/\d{2}(?=\/.*\/\?coord=)"
    label_filter = dates.strftime("%Y%m%d")
    prods = crawl(days, regex, filter_regex, session, label_filter = label_filter)

    regex = ".*\.HDF5"
    filter_regex = "\d{8}(?=.*\.HDF5)"
    fns = crawl(prods, regex, filter_regex, session, list_out = True)

    dl_urls = [os.path.join(re.sub("\?coord=.*","",prods[date_str]), fn) for date_str, fn, in fns]

    log.info(f"--> Downloading {len(dl_urls)} PROBAV tiles.")

    return dl_urls

if __name__ == "__main__":

    product_name = "S5_TOC_100_m_C1"

    folder = r"/Users/hmcoerver/Downloads/pywapor_test"
    # latlim = [26.9, 33.7]
    # lonlim = [25.2, 37.2]
    latlim = [28.9, 29.7]
    lonlim = [30.2, 31.2]
    timelim = [datetime.date(2020, 7, 2), datetime.date(2020, 7, 9)]

    variables = None
    post_processors = None
    req_vars = ["r0", "ndvi"]

    # ds = download(folder, latlim, lonlim, timelim, product_name, req_vars = req_vars,
    #             variables = variables, post_processors = post_processors)

