import xarray as xr
import numpy as np
import os
import rasterio
import rioxarray.merge
import tempfile
from pydap.cas.urs import setup_session
import urllib.parse
from pywapor.general.logger import log
from rasterio.crs import CRS
from pywapor.general.processing_functions import save_ds, process_ds, remove_ds
import warnings
from pywapor.enhancers.apply_enhancers import apply_enhancer
from pywapor.collect.protocol.crawler import download_url, download_urls

def download(fp, product_name, coords, variables, post_processors, 
                fn_func, url_func, un_pw = None, tiles = None,  
                data_source_crs = None, parallel = False, spatial_tiles = True, 
                request_dims = True, timedelta = None):
    """Download data from a OPENDaP server.

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
    un_pw : tuple, optional
        Username and password to use, by default None.
    tiles : list, optional
        Tiles to download, by default None.
    data_source_crs : rasterio.CRS.crs, optional
        CRS of datasource, by default None.
    parallel : bool, optional
        Download files in parallel (currently not implemented), by default False.
    spatial_tiles : bool, optional
        Whether the tiles are spatial or temporal, by default True.
    request_dims : bool, optional
        Include dimension settings in the OPENDaP request, by default True.
    timedelta : datetime.datetime.timedelta, optional
        Shift the time coordinates by tdelta, by default None.

    Returns
    -------
    xr.Dataset
        Dataset with downloaded data.
    """

    folder = os.path.split(fp)[0]

    # Create selection object.
    selection = create_selection(coords, target_crs = data_source_crs)

    # Make output filepaths, should be same length as `urls`.
    fps = [os.path.join(folder, fn_func(product_name, x)) for x in tiles]

    # Make data request URLs.
    session = start_session(url_func(product_name, tiles[0]), selection, un_pw)
    if spatial_tiles:
        idxss = [find_idxs(url_func(product_name, x), selection, session) for x in tiles]
        urls = [create_url(url_func(product_name, x), idxs, variables, request_dims = request_dims) for x, idxs in zip(tiles, idxss)]
    else:
        idxs = find_idxs(url_func(product_name, tiles[0]), selection, session)
        urls = [create_url(url_func(product_name, x), idxs, variables, request_dims = request_dims) for x in tiles]

    # Download data.
    files = download_urls(urls, "", session, fps = fps, parallel = parallel)

    # Merge spatial tiles.
    coords_ = {k: [v[0], selection[v[0]]] for k,v in coords.items()}
    if spatial_tiles:
        dss = [process_ds(xr.open_dataset(x, decode_coords = "all"), coords_, variables, crs = data_source_crs) for x in files]
        ds = rioxarray.merge.merge_datasets(dss)
    else:
        dss = files
        ds = xr.concat([xr.open_dataset(x, decode_coords="all") for x in files], dim = "time")
        ds = process_ds(ds, coords_, variables, crs = data_source_crs)

    # Reproject if necessary.
    if ds.rio.crs.to_epsg() != 4326:
        ds = ds.rio.reproject(CRS.from_epsg(4326))

    ds = ds.rio.clip_box(coords["x"][1][0], coords["y"][1][0], coords["x"][1][1], coords["y"][1][1])

    # Apply product specific functions.
    for var, funcs in post_processors.items():
        for func in funcs:
            ds, label = apply_enhancer(ds, var, func)
            log.info(label)

    if isinstance(timedelta, np.timedelta64):
        ds["time"] = ds["time"] + timedelta

    # Remove unrequested variables.
    ds = ds[list(post_processors.keys())]
    
    # Save final output.
    ds.attrs = {}
    for var in ds.data_vars: # NOTE Keeping band attributes can cause problems when 
        # opening the data using rasterio (in reproject_chunk), see https://github.com/rasterio/rasterio/discussions/2751
        ds[var].attrs = {}

    ds = save_ds(ds, fp, encoding = "initiate", label = "Saving merged data.")

    # Remove temporary files.
    if not isinstance(dss, type(None)):
        for x in dss:
            remove_ds(x)

    return ds

def find_idxs(base_url, selection, session, verify = True):
    def _find_idxs(ds, k, search_range):
        all_idxs = np.where((ds[k] >= search_range[0]) & (ds[k] <= search_range[1]))[0]
        return [np.min(all_idxs), np.max(all_idxs)]
    fp = tempfile.NamedTemporaryFile(suffix=".nc").name
    url_coords = base_url + urllib.parse.quote(",".join(selection.keys()))
    fp = download_url(url_coords, fp, session, waitbar = False, verify = verify)
    ds = xr.open_dataset(fp, decode_coords = "all")
    idxs = {k: _find_idxs(ds, k, v) for k, v in selection.items()}
    return idxs

def create_url(base_url, idxs, variables, request_dims = True):
    if request_dims:
        dims = [f"{k}[{v[0]}:{v[1]}]" for k, v in idxs.items()]
    else:
        dims = []
    varis = [f"{k}{''.join([f'[{idxs[dim][0]}:{idxs[dim][1]}]' for dim in v[0]])}" for k, v in variables.items()]
    url = base_url + urllib.parse.quote(",".join(dims + varis))
    return url

def start_session(base_url, selection, un_pw = [None, None]):
    if un_pw == [None, None]:
        warnings.filterwarnings("ignore", "password was not set. ")
    url_coords = base_url + urllib.parse.quote(",".join(selection.keys()))
    session = setup_session(*un_pw, check_url = url_coords, verify = False)
    return session

def download_xarray(url, fp, coords, variables, post_processors, 
                    data_source_crs = None, timedelta = None):
    """Download a OPENDaP dataset using xarray directly.

    Parameters
    ----------
    url : str
        URL to dataset.
    fp : str
        Path to file to download into.
    coords : dict
        Coordinates to request.
    variables : dict
        Variables to request.
    post_processors : dict
        Additional functions to apply to variables.
    data_source_crs : rasterio.CRS.crs, optional
        CRS of the data source, by default None.
    timedelta : datetime.datetime.timedelta, optional
        Shift the time coordinates by tdelta, by default None.

    Returns
    -------
    xr.Dataset
        Downloaded dataset.
    """

    warnings.filterwarnings("ignore", category=xr.SerializationWarning)
    online_ds = xr.open_dataset(url, decode_coords="all")
    # warnings.filterwarnings("default", category=xr.SerializationWarning)

    # Define selection.
    selection = create_selection(coords, target_crs = data_source_crs)

    # Make the selection on the remote.
    online_ds = online_ds.sel({k: slice(*v) for k, v in selection.items()})

    # Rename variables and assign crs.
    online_ds = process_ds(online_ds, coords, variables, crs = data_source_crs)

    # Download the data.
    ds = save_ds(online_ds, fp.replace(".nc", "_temp.nc"), label = f"Downloading data.")

    # Apply product specific functions.
    for var, funcs in post_processors.items():
        for func in funcs:
            ds, label = apply_enhancer(ds, var, func)
            log.info(label)

    if isinstance(timedelta, np.timedelta64):
        ds["time"] = ds["time"] + timedelta

    # Save final output
    out = save_ds(ds, fp, encoding = "initiate", label = "Saving netCDF.")

    remove_ds(ds)

    return out

def create_selection(coords, target_crs = None, source_crs = CRS.from_epsg(4326)):
    """Create a dictionary that can be given to `xr.Dataset.sel`.

    Parameters
    ----------
    coords : dict
        Dictionary describing the different dimensions over which to select. Possible keys
        are "x" for latitude, "y" for longitude and "t" for time, but other selections
        keys are also allowed (e.g. so select a band). Values are tuples with the first
        value the respective dimension names in the `ds` and the second value the selector.
    target_crs : rasterio.crs.CRS, optional
        crs of the dataset on which the selection will be applied, by default None.
    source_crs : rasterio.crs.CRS, optional
        crs of the `x` and `y` limits in `coords`, by default `epsg:4326`.

    Returns
    -------
    dict
        Dimension names with slices to apply to each dimension.
    """
    selection = {}

    if not isinstance(target_crs, type(None)):
        bounds = rasterio.warp.transform_bounds(source_crs, target_crs, 
                                                coords["x"][1][0], 
                                                coords["y"][1][0], 
                                                coords["x"][1][1], 
                                                coords["y"][1][1])
    else:
        bounds = [coords["x"][1][0], coords["y"][1][0], coords["x"][1][1], coords["y"][1][1]]
    
    selection[coords["x"][0]] = [bounds[0], bounds[2]]
    selection[coords["y"][0]] = [bounds[1], bounds[3]]

    if "t" in coords.keys():
        selection[coords["t"][0]] = [np.datetime64(t) for t in coords["t"][1]]

    for name, lim in coords.values():
        if name not in selection.keys():
            selection[name] = lim

    return selection