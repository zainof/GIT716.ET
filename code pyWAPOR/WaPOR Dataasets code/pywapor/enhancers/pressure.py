def pa_to_kpa(ds, var):
    """Convert Pa to kPa.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to adjust.
    var : str
        Variable in `ds` to adjust.

    Returns
    -------
    xr.Dataset
        Dataset in which `var` has been divided by 1000.
    """
    ds[var] = ds[var] / 1000
    ds[var].attrs = {}
    return ds