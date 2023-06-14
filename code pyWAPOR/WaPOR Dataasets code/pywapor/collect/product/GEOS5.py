from pywapor.collect.protocol import opendap
from pywapor.collect.protocol.projections import get_crss
import os
import xarray as xr
from pywapor.general.processing_functions import open_ds, remove_ds, save_ds
from pywapor.enhancers.temperature import kelvin_to_celsius
from functools import partial
import numpy as np
from pywapor.enhancers.pressure import pa_to_kpa
import copy

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
        "inst3_2d_asm_Nx": {
                    "t2m": [("time", "lat", "lon"), "t_air"],
                    "u2m": [("time", "lat", "lon"), "u2m"],
                    "v2m": [("time", "lat", "lon"), "v2m"],
                    "qv2m": [("time", "lat", "lon"), "qv"],
                    "tqv": [("time", "lat", "lon"), "wv"],
                    "ps": [("time", "lat", "lon"), "p_air"],
                    "slp": [("time", "lat", "lon"), "p_air_0"],
                        },
    }

    req_dl_vars = {
        "inst3_2d_asm_Nx": {
            "t_air": ["t2m"],
            "t_air_max": ["t2m"],
            "t_air_min": ["t2m"],
            "u2m": ["u2m"],
            "v2m": ["v2m"],
            "qv": ["qv2m"],
            "wv": ["tqv"],
            "p_air": ["ps"],
            "p_air_0": ["slp"],
        }
    }

    out = {val:variables[product_name][val] for sublist in map(req_dl_vars[product_name].get, req_vars) for val in sublist}

    return out

def default_post_processors(product_name, req_vars):
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
        "inst3_2d_asm_Nx": {
            "t_air": [kelvin_to_celsius], 
            "t_air_max": [partial(kelvin_to_celsius, in_var = "t_air", out_var = "t_air_max")],
            "t_air_min": [partial(kelvin_to_celsius, in_var = "t_air", out_var = "t_air_min")],
            "u2m": [],
            "v2m": [],
            "qv": [],
            "wv": [],
            "p_air": [pa_to_kpa],
            "p_air_0": [pa_to_kpa],
        }
    }

    out = {k:v for k,v in post_processors[product_name].items() if k in req_vars}

    return out

def download(folder, latlim, lonlim, timelim, product_name, req_vars,
                 variables = None, post_processors = None):
    """Download GEOS5 data and store it in a single netCDF file.

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
    folder = os.path.join(folder, "GEOS5")

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

    spatial_buffer = True
    if spatial_buffer:
        latlim = [latlim[0] - 0.25, latlim[1] + 0.25]
        lonlim = [lonlim[0] - 0.3125, lonlim[1] + 0.3125]

    coords = {"x": ["lon", lonlim], "y": ["lat", latlim], "t": ["time", timelim]}

    if isinstance(variables, type(None)):
        variables = default_vars(product_name, req_vars)

    if isinstance(post_processors, type(None)):
        post_processors = default_post_processors(product_name, req_vars)
    else:
        default_processors = default_post_processors(product_name, req_vars)
        post_processors = {k: {True: default_processors[k], False: v}[v == "default"] for k,v in post_processors.items() if k in req_vars}

    data_source_crs = get_crss("WGS84")

    url = f"https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/assim/{product_name}"

    timedelta = np.timedelta64(90, "m")

    ds = opendap.download_xarray(url, fn, coords, variables, post_processors, 
                                    data_source_crs = data_source_crs,
                                    timedelta = timedelta)

    return ds[req_vars_orig]

if __name__ == "__main__":

    import datetime

    folder = r"/Users/hmcoerver/Downloads/pywapor_test"
    # latlim = [26.9, 33.7]
    # lonlim = [25.2, 37.2]
    latlim = [28.9, 29.7]
    lonlim = [30.2, 31.2]
    timelim = [datetime.date(2020, 7, 1), datetime.date(2020, 7, 11)]

    # GEOS5.
    product_name = "inst3_2d_asm_Nx"

    variables = None
    post_processors = None

    req_vars = [
                "t_air", "u2m", "v2m", "qv", 
                "wv", 
                "p_air", "p_air_0"
                ]

    ds = download(folder, latlim, lonlim, timelim, product_name, req_vars = req_vars)
    print(ds.rio.crs, ds.rio.grid_mapping)