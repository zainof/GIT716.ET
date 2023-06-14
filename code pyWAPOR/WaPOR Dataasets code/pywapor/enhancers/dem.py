import numpy as np
import pywapor.general.processing_functions as pf
import xarray as xr
from pywapor.general.logger import log

def calc_slope(ds, var):
    """Calculate the gradient of `z`.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing var.
    var : str
        Variable name.

    Returns
    -------
    xr.Dataset
        Enhanced dataset.
    """
    reqs = ["z", "y", "x"]
    if np.all([x in ds.variables for x in reqs]):

        dlat, dlon = pf.calc_dlat_dlon(None, None, None, (ds.y.values, ds.x.values))            
        new_x_coords = np.append([0], np.cumsum(np.repeat(np.nanmean(dlon), ds["z"].sizes["x"]-1)))
        new_y_coords = np.append([0], np.cumsum(np.repeat(np.nanmean(dlat), ds["z"].sizes["y"]-1)))
        temp_ds = ds.drop_vars(["x", "y"]).assign_coords({"x": new_x_coords, "y": new_y_coords})

        y = temp_ds["z"].differentiate("x")
        x = temp_ds["z"].differentiate("y")

        y = xr.where((y >= -1e-10) & (y <= 1e-10), 0, y)
        x = xr.where((x >= -1e-10) & (x <= 1e-10), 0, x)

        hypot = np.hypot(x,y)
        slope = np.arctan(hypot) * 180.0 / np.pi

        ds[var] = slope.drop_vars(["x", "y"])
    else:
        log.warning(f"--> Couldn't calculate `{var}`, `{'` and `'.join([x for x in reqs if x not in ds.data_vars])}` is missing.")

    return ds

def calc_aspect(ds, var):
    """Calculate the aspect of `z`.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing var.
    var : str
        Variable name.

    Returns
    -------
    xr.Dataset
        Enhanced dataset.
    """
    reqs = ["z", "y", "x"]
    if np.all([x in ds.variables for x in reqs]):

        dlat, dlon = pf.calc_dlat_dlon(None, None, None, (ds.y.values, ds.x.values))            
        new_x_coords = np.append([0], np.cumsum(np.repeat(np.nanmean(dlon), ds["z"].sizes["x"]-1)))
        new_y_coords = np.append([0], np.cumsum(np.repeat(np.nanmean(dlat), ds["z"].sizes["y"]-1)))
        temp_ds = ds.drop_vars(["x", "y"]).assign_coords({"x": new_x_coords, "y": new_y_coords})

        y = temp_ds["z"].differentiate("x")
        x = temp_ds["z"].differentiate("y")

        y = xr.where((y >= -1e-10) & (y <= 1e-10), 0, y)
        x = xr.where((x >= -1e-10) & (x <= 1e-10), 0, x)

        aspect = np.arctan2(y/np.nanmean(dlat), -x/np.nanmean(dlon))  * 180.0 / np.pi + 180.0

        ds[var] = aspect.drop_vars(["x", "y"])
    else:
        log.warning(f"--> Couldn't calculate `{var}`, `{'` and `'.join([x for x in reqs if x not in ds.data_vars])}` is missing.")

    return ds
