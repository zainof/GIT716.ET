import numpy as np

def adjust_wind_height(ds, var, z = 10):
    """Adjust wind height from `z` to 2m according to FAO56 equation47.

    Parameters
    ----------
    ds : xr.Dataset
        Input
    var : str
        Variable to adjust.
    z : int, optional
        wind height in `ds`, by default 10.

    Returns
    -------
    xr.Dataset
        Dataset in which `var` has been adjusted.
    """

    if "2m" in var:
        old_name = var.replace("2m", f"{z}m")
    else:
        old_name = var
    ds[var] = ds[old_name] * ((4.87)/(np.log(67.8*z - 5.42)))
    return ds

def windspeed(ds, var):
    """Calculate windspeed from `u` and `v` wind vectors.

    Parameters
    ----------
    ds : xr.Dataset
        Input.
    var : str
        Variable in which to store the windspeed.

    Returns
    -------
    xr.Dataset
        Dataset with added `var`.
    """
    if np.all([x in ds.data_vars for x in ["u10m", "v10m"]]):
        ds[var] = np.hypot(ds["u10m"], ds["v10m"])
    return ds

