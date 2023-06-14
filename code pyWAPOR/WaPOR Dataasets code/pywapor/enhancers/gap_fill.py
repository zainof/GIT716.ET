from osgeo import gdal
import numpy as np
import xarray as xr

def gap_fill(ds, var, out_var = None, max_search_dist = 8):
    """Apply gdal.FillNoData to every time slice of `var` in `ds`.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing var.
    var : str
        Variable name.
    out_var : str, optional
        New variable name in case `var` should not be overwritten, by default None.
    max_search_dist : int, optional
        Control the maximum size of gaps to be filled, by default 8.

    Returns
    -------
    xr.Dataset
        Gap filled dataset.
    """
    # Create an empty xr.DataArray to store our results.
    new_data = xr.ones_like(ds[var]) * np.nan

    # Since gdal.FillNodata can only handle one band at a time, we have to loop over each scene.
    for t in ds.time:

        # Select the data at time t.
        da_slice = ds[var].sel(time = t)

        # Convert the data to a np.array with numeric no-data-value
        shape = da_slice.shape
        data = np.copy(da_slice.values)
        mask = np.isnan(data)
        ndv = -9999
        data[mask] = ndv

        # Create an in-memory gdal.Dataset.
        driver = gdal.GetDriverByName("MEM")
        gdal_ds = driver.Create('', shape[1], shape[0], 1, gdal.GDT_Float32)
        band = gdal_ds.GetRasterBand(1)
        band.SetNoDataValue(ndv)
        band.WriteArray(data)

        # Pass the gdal.Dataset to the gap filling algorithm.
        _ = gdal.FillNodata(targetBand = band, maskBand = None, 
                            maxSearchDist = max_search_dist, smoothingIterations = 0)

        # Read the results and replace the no-data-values again.
        array = band.ReadAsArray()
        array[array == ndv] = np.nan

        # Release the gdal.Dataset.
        gdal_ds = gdal_ds.FlushCache()

        # Put the results back into the xr.DataArray.
        new_data = xr.where(new_data.time == t, array, new_data)

    if not isinstance(out_var, type(None)):
        ds[out_var] = new_data
    else: 
        ds[var] = new_data

    return ds