def drop_empty_times(ds, x, drop_vars = ["lst"], *args):
    """Checks for time coordinates at which there is no valid to for any of the variables in
    `drop_vars` and removes those coordinates.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to check.
    x : None
        Not used.
    drop_vars : list, optional
        Variables to check for valid data, by default ["lst"].

    Returns
    -------
    xr.Dataset
        Dataset with adjusted time coordinates.
    """
    for drop_var in drop_vars:
        ds = ds.isel(time = (ds[drop_var].count(dim=["y", "x"]) > 0).values)
    return ds