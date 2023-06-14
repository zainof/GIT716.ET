import os
import tqdm
from pywapor.general.processing_functions import save_ds, open_ds, process_ds, remove_ds
from osgeo import gdal
import urllib
from pywapor.enhancers.apply_enhancers import apply_enhancer
from pywapor.general.logger import log

def download(fp, product_name, coords, variables, post_processors, url_func, 
                gdal_config_options = {}, waitbar = True, ndv = -9999):
    """Download data from a Cloud Optimized Geotiff hosted on a remote server.

    Parameters
    ----------
    fp : str
        Path to file in which to download.
    product_name : str
        Name of product.
    coords : dict
        Coordinate names and boundaries.
    variables : dict
        Keys are variable names, values are additional settings.
    post_processors : dict
        Processors to apply to specific variables.
    url_func : function
        Function that takes `product_name` as input and return a url.
    gdal_config_options : dict, optional
        Additional options passed to `gdal.SetConfigOption`, by default {}.
    waitbar : bool, optional
        Show a download progress bar or not, by default True.
    ndv : int, optional
        No data value to use in new file, by default -9999.

    Returns
    -------
    xr.Dataset
        Dataset with the downloaded data.
    """

    folder, fn = os.path.split(fp)

    # Create folder.
    if not os.path.isdir(folder):
        os.makedirs(folder)

    for k, v in gdal_config_options.items():
        gdal.SetConfigOption(k, v)

    bands = [int(k.replace("Band", "")) for k in variables.keys() if "Band" in k]

    # Define bounding-box.
    bb = [coords["x"][1][0], coords["y"][1][1], 
            coords["x"][1][1], coords["y"][1][0]]

    options_dict = {
            "projWin": bb,
            "format": "netCDF",
            "creationOptions": ["COMPRESS=DEFLATE", "FORMAT=NC4"],
            "noData": ndv,
            "bandList": bands,
    }

    if waitbar:
        # Create waitbar.
        waitbar = tqdm.tqdm(position = 0, total = 100, bar_format='{l_bar}{bar}|', delay = 20)

        # Define callback function for waitbar progress.
        def _callback_func(info, *args):
            waitbar.update(info * 100 - waitbar.n)

        # Add callback to gdal.Translate options.
        options_dict.update({"callback": _callback_func})

    # Set gdal.Translate options.
    options = gdal.TranslateOptions(**options_dict)

    # Check if url is local or online path.
    url = url_func(product_name)
    is_not_local = urllib.parse.urlparse(url).scheme in ('http', 'https',)
    if is_not_local:
        url = f"/vsicurl/{url}"

    # Check if path is free.
    temp_path = fp.replace(".nc", "_temp.nc")
    while os.path.isfile(temp_path):
        temp_path = temp_path.replace(".nc", "_.nc")

    # Run gdal.Translate.
    ds_ = gdal.Translate(temp_path, url, options = options)

    # Reset the gdal.Dataset.
    ds_.FlushCache()
    ds_ = None

    # Process the new netCDF.
    ds = open_ds(temp_path)

    # TODO Also check COPERNICUS and GLOBCOVER...
    if int(gdal.__version__.split(".")[0]) < 3 and "WaPOR" in product_name:
        # NOTE this is terrible, but Google Colab uses a 7 year old GDAL version in 
        # which gdal.Translate does not apply scale-factors...
        log.warning("--> You are using a very old GDAL version, applying a `0.01` scale factor manually.")
        ds = ds * 0.01

    ds = ds.rename_vars({k: f"Band{v}" for k,v in zip(list(ds.data_vars), bands)})

    ds = process_ds(ds, coords, variables)

    # Apply product specific functions.
    for var, funcs in post_processors.items():
        for func in funcs:
            ds, label = apply_enhancer(ds, var, func)
            # log.info(label)

    # Save final output.
    out = save_ds(ds, fp, encoding = "initiate", label = f"Saving {fn}.")

    # Remove the temporary file.
    remove_ds(ds)

    return out
