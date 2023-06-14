# https://gitlab.ssec.wisc.edu/sips/OrbNavClient
# https://sips.ssec.wisc.edu/orbnav#/

import json
import fnmatch
import os
import datetime
import glob
import warnings
from functools import partial
import geopy.distance
import xarray as xr
import pandas as pd
import numpy as np
import hashlib
import json
from datetime import datetime as dt
from itertools import chain
import copy
from pywapor.general.logger import log
from pywapor.collect import accounts
from pywapor.enhancers.apply_enhancers import apply_enhancer
from pywapor.general.processing_functions import save_ds, open_ds, remove_ds, adjust_timelim_dtype
from pywapor.general.curvilinear import create_grid, regrid
from pywapor.collect.protocol.crawler import download_url, download_urls
from pywapor.general.logger import log, adjust_logger
from pywapor.enhancers.other import drop_empty_times

def regrid_VNP(workdir, latlim, lonlim, dx_dy = (0.0033, 0.0033)):
    """Process VIIRS images from curvilinear to rectolinear grid, applies cloud masks, quality masks 
    and convert data to Brightness Temperature.

    Parameters
    ----------
    workdir : str
        Folder to search for scenes.
    latlim : list
        Latitude limits of area of interest.
    lonlim : list
        Longitude limits of area of interest.
    dx_dy : tuple, optional
        Output pixel size, by default (0.0033, 0.0033).

    Returns
    -------
    xr.Dataset
        Output dataset.
    """

    # Reformat bounding-box.
    bb = [lonlim[0], latlim[0], lonlim[1], latlim[1]]

    # Search for VNP02 images in workdir.
    ncs02 = glob.glob(os.path.join(workdir, "VNP02IMG.A*.nc"))

    # Make inventory of complete scenes in workdir.
    scenes = dict()
    for nc02 in ncs02:
        dt_str = ".".join(os.path.split(nc02)[-1].split(".")[1:3])
        ncs03 = glob.glob(os.path.join(workdir, f"VNP03IMG.{dt_str}*.nc"))
        ncs_cloud = glob.glob(os.path.join(workdir, f"CLDMSK_L2_VIIRS_SNPP.{dt_str}*.nc"))
        if len(ncs03) != 1 or len(ncs_cloud) != 1:
            continue
        else:
            nc03 = ncs03[0]
            nc_cloud = ncs_cloud[0]
        dt_obj = dt.strptime(dt_str, "A%Y%j.%H%M")
        scenes[dt_obj] = (nc02, nc03, nc_cloud)

    # Create list to store xr.Datasets from a single scene.
    dss = list()

    log.info(f"--> Processing {len(scenes)} scenes.").add()

    # Loop over the scenes.
    for i, (dt_obj, (nc02, nc03, nc_cloud)) in enumerate(scenes.items()):

        # Define tile output path.
        fp = os.path.join(workdir, f"VNP_{dt_obj:%Y%j%H%M}.nc")

        if os.path.isfile(fp):
            dss.append(open_ds(fp))
            continue

        # Open the datasets.
        ds1 = xr.open_dataset(nc02, group = "observation_data", decode_cf = False)
        ds2 = xr.open_dataset(nc03, group = "geolocation_data", decode_cf = False, chunks = "auto")
        ds3 = xr.open_dataset(nc_cloud, group = "geophysical_data", decode_cf = False, chunks = "auto")

        # Create cloud mask. (0=cloudy, 1=probably cloudy, 2=probably clear, 3=confident clear, -1=no result)
        cmask = ds3.Integer_Cloud_Mask.interp({
                        "number_of_lines": np.linspace(0-0.25, (ds3.number_of_lines.size-1)+0.25, ds3.number_of_lines.size*2),
                        "number_of_pixels": np.linspace(0-0.25, (ds3.number_of_pixels.size-1)+0.25, ds3.number_of_pixels.size*2)
                        }, kwargs = {"fill_value": "extrapolate"}).drop_vars(["number_of_lines", "number_of_pixels"])

        # Convert DN to Brightness Temperature using the provided loopup table.
        bt_da = ds1.I05_brightness_temperature_lut.isel(number_of_LUT_values = ds1.I05)

        # Rename some things.
        ds = ds2[["latitude", "longitude"]].rename({"latitude": "y", "longitude": "x"})

        # Chunk and mask invalid pixels.
        ds["bt"] = bt_da.chunk("auto").where((bt_da >= bt_da.valid_min) & 
                                            (bt_da <= bt_da.valid_max) & 
                                            (bt_da != bt_da._FillValue) &
                                            (ds1.I05_quality_flags == 0) &
                                            (ds2.quality_flag == 0) &
                                            (cmask == 3)
        )

        # Move `x` and `y` from variables to coordinates
        ds = ds.set_coords(["x", "y"])

        # Create mask for selecting the bounding-box in the data.
        buffer = 0.2
        mask = ((ds.y >= bb[1] - buffer) &
                (ds.y <= bb[3] + buffer) & 
                (ds.x >= bb[0] - buffer) & 
                (ds.x <= bb[2] + buffer))

        # Apply the mask.
        ds = ds.where(mask.compute(), drop = True)

        # Stop with scene if no valid data is found inside bb.
        if ds.bt.count().values == 0:
            log.warning(f"--> ({i+1}/{len(scenes)}) Skipping scene `{os.path.split(fp)[-1]}`, no valid data found inside bounding-box.")
            continue

        # Save intermediate file.
        fp_temp = os.path.join(workdir, f"VNP_{dt_obj:%Y%j%H%M}_temp.nc")
        ds = save_ds(ds, fp_temp, label = f"({i+1}/{len(scenes)}) Creating intermediate file.")

        # Create rectolinear grid.
        grid_ds = create_grid(ds, dx_dy[0], dx_dy[1], bb = bb)

        # Regrid from curvilinear to rectolinear grid.
        out = regrid(grid_ds, ds, max_px_dist = 3)

        # Set some metadata.
        out = out.rio.write_crs(4326)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category = FutureWarning)
            out = out.rio.clip_box(*bb)
        out.bt.attrs = {k: v for k, v in ds.bt.attrs.items() if k in ["long_name", "units"]}

        # Add time dimension.
        out = out.expand_dims({"time": 1}).assign_coords({"time": [dt_obj]})

        # Save regridded tile.
        out = save_ds(out, fp, encoding = "initiate", label = f"({i+1}/{len(scenes)}) Regridding `VNP_{dt_obj:%Y%j%H%M}.nc` to rectolinear grid.")

        dss.append(out)

        # Remove intermediate file.
        remove_ds(ds)

    log.sub()

    ds = xr.merge(dss)

    return ds

def boxtimes(latlim, lonlim, timelim, folder):
    """Check in which 6-min periods SUOMI NPPs swatch crosses a AOI
    defined by `latlim` and `lonlim` and remove night recordings.

    Parameters
    ----------
    latlim : list
        Latitude limits.
    lonlim : list
        Longitude limits.
    timelim : list
        Time limits.

    Returns
    -------
    list
        List containing year, doy and time at which SUOMI NPP passes over
        the bb during the day (night is filtered out).
    """

    # Convert to datetime object if necessary
    timelim = adjust_timelim_dtype(timelim)

    # Expand bb with 1500km (is half of SUOMI NPP swath width).
    ur = (latlim[1], lonlim[1])
    ll = (latlim[0], lonlim[0])
    very_ur = geopy.distance.distance(kilometers=1500).destination(ur, bearing=45)
    very_ll = geopy.distance.distance(kilometers=1500).destination(ll, bearing=45+180)

    # Define search kwargs.
    kwargs = {
        "sat" : '37849',
        "start" : timelim[0].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end" : timelim[1].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ur" : f"{int(np.ceil(very_ur.latitude))},{int(np.ceil(very_ur.longitude))}",
        "ll" : f"{int(np.floor(very_ll.latitude))},{int(np.floor(very_ll.longitude))}",
    }

    # Create a hash for the search kwargs to create a filename for this specific search, allowing
    # to reuse the file later if a similar search is made.
    search_hash = hashlib.sha1(json.dumps(kwargs, sort_keys=True).encode()).hexdigest()

    # Search boxtimes.
    base_url = "https://sips.ssec.wisc.edu/orbnav/api/v1/boxtimes.json"
    url = base_url + "?&" + "&".join([f"{k}={v}" for k, v in kwargs.items()])
    fp = os.path.join(folder, f"boxtimes_{search_hash}.json")
    _ = download_url(url, fp)
    with open(fp) as f:
        data = json.load(f)

    # Group to SUOMI NPP 6 minute tiles.
    dates = [dt.strptime(x[0], "%Y-%m-%dT%H:%M:%SZ") for x in chain.from_iterable(data["data"]) if x[3] > 40.]
    df = pd.DataFrame({"date": dates}).set_index("date")
    count = df.groupby(pd.Grouper(freq = "6min")).apply(lambda x: len(x))
    to_dl = count[count >= 1].index.values

    # Create list with relevant year/doy/times tuples.
    year_doy_time = [[pd.to_datetime(x).strftime("%Y"), pd.to_datetime(x).strftime("%j"), pd.to_datetime(x).strftime("%H%M")] for x in to_dl]

    return year_doy_time

def find_VIIRSL1_urls(year_doy_time, product, workdir, 
                        server_folders = {"VNP02IMG": 5110, "VNP03IMG": 5200, "CLDMSK_L2_VIIRS_SNPP": 5110}):
    """Given a list of year/doy/time tuples returns the exact urls for that
    datetime. Also check if the linked file already exists in workdir, in which
    case the url is omitted from the returned list.

    Parameters
    ----------
    year_doy_time : list
        List containing year, doy and time at which SUOMI NPP passes over
        the bb during the day (night is filtered out).
    product : str
        Name of the product to search.
    workdir : str
        Path to folder in which to store data.
    server_folder : dict
        Which server side folder to use per product, by default {"VNP02IMG": 5110, "VNP03IMG": 5200, "CLDMSK_L2_VIIRS_SNPP": 5110}.

    Returns
    -------
    list
        URLs linking to the product.
    """
    # Make empty url list.
    urls = list()

    # Loop over year/doy/times.
    for year, doy, t in year_doy_time:

        # Define url at which to find the json with the exact tile urls.
        url = f"https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/{server_folders[product]}/{product}/{year}/{doy}.json"

        # Download the json.
        fp = os.path.join(workdir, f"{product}_{year}{doy}.json")
        if not os.path.isfile(fp):
            fp = download_url(url, fp)

        # Open the json and find the exact url of the required scene.
        all_scenes = [x["name"] for x in json.load(open(fp))["content"]]
        req_scenes = fnmatch.filter(all_scenes, f"{product}.A{year}{doy}.{t}.*.*.nc")

        # Check if a corrct url is found.
        if len(req_scenes) == 1:
            fn = req_scenes[0]
        else:
            continue

        # Check if the file already exists in workdir, otherwise add it to `urls`.
        url = f"https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/{server_folders[product]}/{product}/{year}/{doy}/{fn}"
        fp = os.path.join(workdir, fn)
        if os.path.exists(fp):
            continue
        urls.append(url)

    return urls

def check_tiles(year_doy_time, latlim, lonlim, workdir, product = "VNP03IMG"):
    """Check if downloaded tiles actually contain data for the given AOI. If not
    removes its year/doy/time from the list.

    Parameters
    ----------
    year_doy_time : list
        List containing year, doy and time at which SUOMI NPP passes over
        the bb during the day (night is filtered out).
    latlim : list
        Latitude limits.
    lonlim : list
        Longitude limits.
    workdir : str
        Path to working directory
    product : str, optional
        name of the product to check, by default "VNP03IMG"

    Returns
    -------
    tuple
        Two lists, the first an updated `year_date_time`, the other contains the 
        values dropped from `year_date_time`.
    """
    # Create empty list to store dropped `year/doy/times`.
    dropped = list()

    # Reformat bounding-box.
    bb = [lonlim[0], latlim[0], lonlim[1], latlim[1]]

    # Loop over `year/doy/times`.
    for year, doy, time in year_doy_time:

        # Search for the relevant .nc file.
        fps = glob.glob(os.path.join(workdir, f"{product}.A{year}{doy}.{time}.*.nc"))
        assert len(fps) == 1
        fp = fps[0]

        # Open the geolocation data.
        ds = xr.open_dataset(fp, group = "geolocation_data", chunks = "auto")

        # Mask the irrelevant pixels.
        buffer = 0.2
        mask = ((ds.latitude >= bb[1] - buffer) &
                (ds.latitude <= bb[3] + buffer) & 
                (ds.longitude >= bb[0] - buffer) & 
                (ds.longitude <= bb[2] + buffer))

        # Check if there is relevant data in this tile.
        if mask.sum().values < 100:
            year_doy_time.remove([year, doy, time])
            dropped.append([year, doy, time])

    return year_doy_time, dropped

def check_geoloc_tile(fp, bb, min_pixels = 100):
    """Checks if there is relevant data in the dataset.

    Parameters
    ----------
    fp : str
        Path to input file.
    bb : list
        Boundingbox, [xmin, ymin, xmax, ymax].
    min_pixels : int, optional
        Threshold value, by default 100.

    Returns
    -------
    bool
        True if there are more than `min_pixels` inside the `bb`.
    """

    # Open the geolocation data.
    try: 
        ds = xr.open_dataset(fp, group = "geolocation_data", chunks = "auto")
    except OSError as e:
        if "HDF error" in str(e):
            log.info(f"--> {os.path.split(fp)[-1]} is corrupted, removing it.")
            os.remove(fp)
            return None
        else:
            raise e

    # Mask the irrelevant pixels.
    buffer = 0.2
    mask = ((ds.latitude >= bb[1] - buffer) &
            (ds.latitude <= bb[3] + buffer) & 
            (ds.longitude >= bb[0] - buffer) & 
            (ds.longitude <= bb[2] + buffer))

    # Check if there is relevant data in this tile.
    return mask.sum().values > min_pixels

def check_nc(fp, group = "observation_data"):

    try: 
        _ = xr.open_dataset(fp, group = group, chunks = "auto")
    except OSError as e:
        if "HDF error" in str(e):
            log.info(f"--> {os.path.split(fp)[-1]} is corrupted, removing it.")
            os.remove(fp)
            return None
        else:
            raise e

    return True

def default_vars(product_name, req_vars):
    """Given a `product_name` and a list of requested variables, returns a dictionary
    with metadata on which exact layers need to be requested from the server, how they should
    be renamed, and how their dimensions are defined.

    Parameters
    ----------
    product_name : str
        Name of the product.
    req_vars : list
        List of variables to be collected.

    Returns
    -------
    dict
        Metadata on which exact layers need to be requested from the server.
    """

    variables = {
        "VNP02IMG": {"bt": {(), "bt"}},
    }

    req_dl_vars = {
        "VNP02IMG": {
            "bt": ["bt"],
        },
    }

    out = {val:variables[product_name][val] for sublist in map(req_dl_vars[product_name].get, req_vars) for val in sublist}
    
    return out

def default_post_processors(product_name, req_vars = None):
    """Given a `product_name` and a list of requested variables, returns a dictionary with a 
    list of functions per variable that should be applied after having collected the data
    from a server.

    Parameters
    ----------
    product_name : str
        Name of the product.
    req_vars : list
        List of variables to be collected.

    Returns
    -------
    dict
        Functions per variable that should be applied to the variable.
    """
    post_processors = {
        "VNP02IMG": {
            "bt": [partial(drop_empty_times, drop_vars = ["bt"])]
            },
    }

    out = {k:v for k,v in post_processors[product_name].items() if k in req_vars}

    return out

def download(folder, latlim, lonlim, timelim, product_name, req_vars,
                variables = None, post_processors = None):
    """Download VIIRSL1 data and store it in a single netCDF file.

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
    req_vars : list
        Which variables to download for the selected product.
    variables : dict, optional
        Metadata on which exact layers need to be requested from the server, by default None.
    post_processors : dict, optional
        Functions per variable that should be applied to the variable, by default None.

    Returns
    -------
    xr.Dataset
        Downloaded data.
    """
    folder = os.path.join(folder, "VIIRSL1")
    if not os.path.exists(folder):
        os.makedirs(folder)

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

    if isinstance(variables, type(None)):
        variables = default_vars(product_name, req_vars)

    if isinstance(post_processors, type(None)):
        post_processors = default_post_processors(product_name, req_vars)
    else:
        default_processors = default_post_processors(product_name, req_vars)
        post_processors = {k: {True: default_processors[k], False: v}[v == "default"] for k,v in post_processors.items() if k in req_vars}

    # Reformat bounding-box.
    bb = [lonlim[0], latlim[0], lonlim[1], latlim[1]]

    # Find SUOMI NPP overpass times.
    year_doy_time = boxtimes(latlim, lonlim, timelim, folder)

    # Search for valid urls at overpass times for geolocation files.
    urls = find_VIIRSL1_urls(year_doy_time, "VNP03IMG", folder)

    # Create authentication header.
    _, token = accounts.get('VIIRSL1')
    headers = {'Authorization': 'Bearer ' + token}

    # Try to download the geolocation data.
    checker_function = partial(check_geoloc_tile, bb = bb)
    relevant_fps = download_urls(urls, folder, headers = headers, checker_function = checker_function)

    # Check relevance of files that were already downloaded.
    fns = [os.path.split(x)[-1] for x in glob.glob(os.path.join(folder, "VNP03IMG*.nc"))]
    for fp in [os.path.join(folder, x) for x in fns if x not in [os.path.split(x)[-1] for x in urls]]:
        if check_geoloc_tile(fp, bb):
            relevant_fps.append(fp)

    # Convert the relevant netcdf names to year/doy/time entries.
    year_doy_time = [[os.path.split(x)[-1][{0:slice(10,14), 1:slice(14,17), 2:slice(18,22)}[y]] for y in range(3)] for x in relevant_fps]

    # Find urls of the cloudmask for the given year/doy/time.
    cloud_urls = find_VIIRSL1_urls(year_doy_time, "CLDMSK_L2_VIIRS_SNPP", folder)
    # Download the cloudmasks.
    _ =  download_urls(cloud_urls, folder, headers = headers, checker_function = partial(check_nc, group = "geophysical_data"))

    # Find urls of the actual data for the given year/doy/time.
    product_urls = find_VIIRSL1_urls(year_doy_time, product_name, folder)
    # Download the data product.
    _ =  download_urls(product_urls, folder, headers = headers, checker_function = check_nc)

    # Combine geolocations and data in rectilinear grid.
    ds = regrid_VNP(folder, latlim, lonlim, dx_dy = (0.0033, 0.0033))

    # Apply product specific functions.
    for var, funcs in post_processors.items():
        for func in funcs:
            ds, label = apply_enhancer(ds, var, func)
            log.info(label)

    ds = save_ds(ds, fn, encoding = "initiate", label = "Merging files.")

    return ds[req_vars_orig]

if __name__ == "__main__":

    folder = "/Users/hmcoerver/Local/20220325_20220415_test_data"
    adjust_logger(True, folder, "INFO")
    # latlim = [28.9, 29.7]
    # lonlim = [30.2, 31.2]
    # timelim = ["2022-06-01", "2022-06-02"]

    timelim = [datetime.date(2022, 3, 25), datetime.date(2022, 4, 15)]
    latlim = [29.4, 29.7]
    lonlim = [30.7, 31.0]

    product_name = "VNP02IMG"
    req_vars = ["bt"]

    variables = None
    post_processors = None

    ds = download(folder, latlim, lonlim, timelim, product_name, req_vars)