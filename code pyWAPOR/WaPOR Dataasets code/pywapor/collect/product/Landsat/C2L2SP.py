"""Functions to process Landsat 7 and 8 Collection-2 Level-2 scenes into
NDVI, ALBEDO and LST GeoTIFFs that can be ingested into pywapor.pre_et_look.
"""
import rasterio
import glob
import os
import tarfile
import json
import shutil
from pywapor.general.processing_functions import save_ds, make_example_ds, remove_ds
import numpy as np
import xarray as xr
from osgeo import gdal
from datetime import datetime
from pywapor.general.logger import log
from pywapor.general import bitmasks
from pywapor.collect.protocol.crawler import download_url

def main(folder, max_lst_uncertainty = 2.5, final_bb = None):
    """Processes Landsat 7 or 8 Collection-2 Level-2 tar-files into GeoTIFFs
    for NDVI, LST and ALBEDO.

    Parameters
    ----------
    folder : str
        Path to folder containing Landsat 7 or 8 Collection-2 Level-2 
        tar-files.
    final_bb : list
        left, bottom, right, top.

    Returns
    -------
    ndvi_files : list
        List of paths to the NDVI files created.
    albedo_files : list
        List of paths to the ALBEDO files created.
    lst_files : list
        List of paths to the LST files created.
    """

    # Look for tar-files in the input folder.
    all_files = glob.glob(os.path.join(folder, "*.tar"))

    # Filter tar-files down to Landsat Level-2 files.
    files = [os.path.split(file)[-1] for file in all_files if "L2SP" in file]

    dss = list()

    example_ds = None

    # Loop over the tar-files.
    for file in files:

        # Unpack tar-file.
        ls_folder = untar(file, folder)

        ## ALBEDO & NDVI ##
        to_open = ["blue", "red", "green", "nir"]
        
        # Open the data.
        data, mtl, proj, geot = open_data(ls_folder, to_open)

        # Make cloud/shadow/quality mask.
        valid = open_mask(ls_folder, mtl, proj, geot)

        # Mask pixels.
        data = data.where(valid)

        # Calculate the albedo.
        albedo = calc_albedo(data)

        # Calculate the NDVI.
        ndvi = calc_ndvi(data)

        ## LST ##
        # Define which bands are required.
        to_open = ["therm", "therm_qa"]

        # Open the data.
        data = open_data(ls_folder, to_open)[0]
        
        # Create mask for pixels with an uncertainty greater than x Kelvin.
        valid = data.sel(band = "therm_qa") <= max_lst_uncertainty * 100

        # Apply mask.
        data = data.where(valid)
        
        # Calculate the lst.
        lst = calc_lst(data)

        # Merge the three variables into one dataset.
        ds = xr.merge([lst, ndvi, albedo])

        # Add time dimension to data arrays.
        ds = ds.expand_dims({"time": 1})

        # Set the correct time.
        date_str = mtl["IMAGE_ATTRIBUTES"]["DATE_ACQUIRED"]
        time_str = mtl["IMAGE_ATTRIBUTES"]["SCENE_CENTER_TIME"]
        datetime_str = date_str + " " + time_str.replace("Z", "")
        ds = ds.assign_coords({"time":[np.datetime64(datetime_str)]})

        target_crs = rasterio.crs.CRS.from_epsg(4326)

        # Clip and pad to bounding-box
        if isinstance(example_ds, type(None)):
            example_ds = make_example_ds(ds, os.path.join(folder, os.path.splitext(file)[0]), target_crs, bb = final_bb)
        ds = ds.rio.reproject_match(example_ds)
        ds = ds.assign_coords({"x": example_ds.x, "y": example_ds.y})

        # Save to netcdf
        fp = os.path.join(folder, os.path.splitext(file)[0] + ".nc")
        ds = save_ds(ds, fp, label = f"Processing `{file}`.")

        dss.append(ds)

        # Remove the untarred files and folder.
        shutil.rmtree(ls_folder)

    ds = xr.merge(dss, combine_attrs = "drop")

    fp = os.path.join(folder, "LANDSAT.nc")

    if os.path.isfile(fp):
        os.remove(fp)

    # encoding = {v: {"zlib": True, "dtype": "float32"} for v in list(ds.data_vars)}
    # encoding["time"] = {"dtype": "float64"}
    ds = ds.rio.write_crs(target_crs)
    
    ds = save_ds(ds, fp, encoding = "initiate", label = "Merging files.")

    for x in dss:
        remove_ds(x)

    return ds

def open_mask(ls_folder, mtl, proj, geot):
    """Create a mask to remove clouds/shadows and oversaturated pixels
    from the scene.

    Parameters
    ----------
    ls_folder : str
        Path to folder containing the scene images.
    mtl : dict
        Metadata of the landsat scene.
    proj : str
        Projection of the images to which the mask will be applied.
    geot : tuple
        Geotransform of the images to which the mask will be applied.

    Returns
    -------
    xr.DataArray
        Mask to apply to surface reflectance bands. True indicates pixels that
        should be kept, False indicates pixels that should be removed.
    """

    # Find Landsat spacecraft number.
    ls_number = ls_number_from_mtl(mtl)

    # Define which general bands to open.
    to_open = ["pixel_qa", "radsat_qa"]

    # Look up the filepaths.
    fps = find_files(ls_folder, to_open)

    # Open the data.
    qa_data = xr.concat([open_as_xr(fp, name) for name, fp in fps.items()], "band")
    
    # Check if projection and geotransform is the same as data to which the 
    # mask will be applied.
    _, _ = check_projs_geots(fps.values(), ref_proj_geot = (proj, geot))
    
    # Open the bit numbers.
    pixel_qa_bits = bitmasks.get_pixel_qa_bits(2, ls_number, 2)
    
    # Choose which labels to mask (see keys of 'pixel_qa_bits' for options).
    if ls_number in [8, 9]:
        pixel_qa_flags = ["dilated_cloud", "cirrus", "cloud", "cloud_shadow", "snow"]
    elif ls_number in [4, 5, 7]:
        pixel_qa_flags = ["dilated_cloud", "cloud", "cloud_shadow", "snow"]
    else:
        raise ValueError

    # Load array into RAM.
    qa_array = qa_data.sel(band="pixel_qa").band_data.values.astype("uint16")
    
    # Create the first mask.
    mask1 = bitmasks.get_mask(qa_array, pixel_qa_flags, pixel_qa_bits)

    # Open the bit numbers
    radsat_qa_bits = bitmasks.get_radsat_qa_bits(2, ls_number, 2)
    
    # Choose which labels to mask (all in this case).
    radsat_qa_flags = list(radsat_qa_bits.keys())
    
    # Load array into RAM.
    qa_array = qa_data.sel(band="radsat_qa").band_data.values.astype("uint16")
    
    # Create the second mask.
    mask2 = bitmasks.get_mask(qa_array, radsat_qa_flags, radsat_qa_bits)
    
    # Combine the masks.
    mask = np.invert(np.any([mask1, mask2], axis = 0))
    
    # Put the mask inside a xr.DataArray.
    mask_xr = xr.DataArray(data = np.transpose(mask), 
                            coords = qa_data.drop_dims(["band"]).coords)

    return mask_xr

def open_data(ls_folder, to_open):
    """Opens data in a folder created by unpacking a Landsat Collection-2 
    Level-2 tar-file, applies scale and offset factors, and masks invalid 
    values.

    Parameters
    ----------
    ls_folder : str
        Path to folder containing the scene images.
    to_open : list
        List with general band-names that need to be opened, e.g. ["red", "nir].

    Returns
    -------
    data : xr.Dataset
        Xarray dataset containing all the requested bands.
    mtl : dict
        Metadata of the landsat scene.
    proj : str
        Projection of the images.
    geot : tuple
        Geotransform of the images
    """

    # Open MTL file.
    mtl = open_mtl(ls_folder)

    # Check Landsat number.
    ls_number = ls_number_from_mtl(mtl)

    # Find the files that need to be opened.
    fps = find_files(ls_folder, to_open)

    # Check if all files have the same proj and geot.
    proj, geot = check_projs_geots(fps.values())

    # Open all the files into an xarray.Dataset.
    data = xr.concat([open_as_xr(fp, name) for name, fp in fps.items()], "band")

    # Find valid data range for each file.
    valid_ranges = valid_range()[ls_number]

    # Mask data outside valid range.
    masked_data = mask_invalid(data, valid_ranges)

    # Find relevant scales and offset factors for each file.
    scales_xr, offsets_xr = find_scales_offsets(fps, mtl)
    
    # Apply scale and offset to data.
    scaled_data = masked_data * scales_xr + offsets_xr
    
    # Add metadata.
    scaled_data.attrs = {"ls_number": ls_number}

    return scaled_data, mtl, proj, geot

def ls_number_from_mtl(mtl):
    """Find which Landsat spacecraft this mtl refers to.

    Parameters
    ----------
    mtl : dict
        Metadata of the landsat scene.

    Returns
    -------
    int
        Either 7 or 8.
    """
    return int(mtl["IMAGE_ATTRIBUTES"]["SPACECRAFT_ID"].split("_")[-1])

def scale_offset_from_mtl(fp, mtl, type):
    """Look up the scale and offset factor for a given filepath.

    Parameters
    ----------
    fp : str
        Path to GeoTIFF of a Landsat band.
    mtl : dict
        Metadata of the landsat scene.
    type : str
        Type of data in the fp, either "TEMPERATURE" or "REFLECTANCE".

    Returns
    -------
    float
        Scale factor to be applied to the fp.
    float
        Offset factor to be applied to the fp.
    """
    bandname = os.path.splitext(fp)[0].split("_")[-1][1:]

    addon = {True: "_ST_B", False: "_"}[type == "TEMPERATURE"]

    try:
        scale = float(mtl[f'LEVEL2_SURFACE_{type}_PARAMETERS'][f'{type}_MULT_BAND{addon}{bandname}'])
        offset = float(mtl[f'LEVEL2_SURFACE_{type}_PARAMETERS'][f'{type}_ADD_BAND{addon}{bandname}'])
    except KeyError:
        # log.warning(f"No scale/offset found for {fp}, setting to 1 and 0.")
        scale = 1.0
        offset = 0.0
    return scale, offset

def find_scales_offsets(fps, mtl):
    """Find scale and offset factors for multiple files and store them in
    xarray.DataArrays.

    Parameters
    ----------
    fps : dict
        Keys are general band-names, values are paths to files.
    mtl : dict
        Metadata of the landsat scene.

    Returns
    -------
    xarray.DataArray
        Array with the scale factors per band.
    xarray.DataArray
        Array with the offset factors per band.
    """

    scales_offsets = dict()
    for k, v in fps.items():
        if "therm" in k:
            scales_offsets[k] = scale_offset_from_mtl(v, mtl, "TEMPERATURE")
        else:
            scales_offsets[k] = scale_offset_from_mtl(v, mtl, "REFLECTANCE")

    scales_xr = xr.DataArray(data = np.array(list(scales_offsets.values()))[:,0], 
                        coords = {"band": list(scales_offsets.keys())})
    offsets_xr = xr.DataArray(data = np.array(list(scales_offsets.values()))[:,1], 
                        coords = {"band": list(scales_offsets.keys())})
    
    return scales_xr, offsets_xr

def proj_geot(fp, ds = None):
    """Find the projection and geotransform of a GeoTIFF file.

    Parameters
    ----------
    fp : str
        Path to GeoTIFF of a Landsat band.
    ds : gdal.Dataset, optional
        Alternatively, can use gdal.Dataset instead of fp, by default None

    Returns
    -------
    str
        Projection of the images.
    tuple
        Geotransform of the images.
    """
    if isinstance(ds, type(None)):
        ds = gdal.Open(fp)
    proj = ds.GetProjection()
    geot = ds.GetGeoTransform()
    return proj, geot

def acquired_at(mtl):
    """Find on which date and time the given scene was aqcuired.

    Parameters
    ----------
    mtl : dict
        Metadata of the landsat scene.

    Returns
    -------
    datetime.datetime
        Aqcuisition date and time.
    """
    date_time_str = mtl["IMAGE_ATTRIBUTES"]["DATE_ACQUIRED"] + mtl["IMAGE_ATTRIBUTES"]["SCENE_CENTER_TIME"]
    date_time = datetime.strptime(date_time_str[:-2], "%Y-%m-%d%H:%M:%S.%f")
    return date_time

def create_fn(var, mtl):
    """Create a filename conform pywapor.pre_et_look input files.

    Parameters
    ----------
    var : str
        Name of the variable for which to create a filename.
    mtl : dict
        Metadata of the landsat scene.

    Returns
    -------
    str
        Filename.
    """
    ls_number = ls_number_from_mtl(mtl)
    date_time = acquired_at(mtl)
    unit = {"lst": "K", "ndvi": "-", "r0": "-"}
    fn = f"{var}_LS{ls_number:01d}_{unit[var]}_-_{date_time:%Y.%m.%d.%H.%M}.tif"
    return fn

def calc_ndvi(data):
    """Calculate the Normalized Difference Vegetation Index (NDVI).

    Parameters
    ----------
    data : xarray.Dataset
        Dataset that needs to contain at least the coordinates "red" and "nir"
        in a dimension called 'band'.

    Returns
    -------
    xarray.Dataset
        Dataset that has a variable called "ndvi".
    """

    red = data.band_data.sel(band = "red")
    nir = data.band_data.sel(band = "nir")

    ndvi = (nir - red) / (nir + red)

    return xr.Dataset({"ndvi": ndvi})

def calc_albedo(data, ls_number = None, weights = None):
    """Calculate the albedo as a weighted average of the surface reflections.

    Parameters
    ----------
    data : xarray.Dataset
        Dataset that needs to contain at least the variables "red", "green",
        "blue", "nir", "swir1" and "swir2".
    ls_number : int, optional
        Landsat spacecraft number, can also be taken from 
        data.attrs["ls_number"], by default None.
    weights : dict, optional
        Dictionary whose keys are the same as the bandnames in data
        and its values the weights to be applied, uses default weights when 
        is None, by default None.

    Returns
    -------
    xarray.Dataset
        Dataset that has a variable called "ndvi".
    """

    if isinstance(weights, type(None)):
        if isinstance(ls_number, type(None)):
            ls_number = data.attrs["ls_number"]
        weights = albedo_weight()[ls_number]

    offset_band = xr.ones_like(data.isel(band=0)).expand_dims("band").assign_coords({"band": ["offset"]})
    offset_band = offset_band.where(data.band_data.isel(band=0).notnull())
    all_data = xr.concat([data, offset_band], "band")

    weights_xr = xr.DataArray(data = list(weights.values()), 
                            coords = {"band": list(weights.keys())})

    albedo = (all_data * weights_xr).sum("band", skipna = False).rename({"band_data": "r0"})

    return albedo

def calc_lst(data):
    """Calculate the land surface temperature.

    Parameters
    ----------
    data : xarray.Dataset
        Dataset that needs to contain at least the coordinate "therm" in the
        'band' dimension.

    Returns
    -------
    xarray.Dataset
        Dataset that has a variable called "lst".
    """
    return data.sel(band="therm").rename({"band_data": "lst"}).drop_vars(["band"])

def open_mtl(ls_folder):
    """Opens the MTL file that is included in every Landsat tar-file.

    Parameters
    ----------
    ls_folder : str
        Path to folder containing the scene images.

    Returns
    -------
    dict
        Metadata of the landsat scene.
    """
    fp_mtl = glob.glob(os.path.join(ls_folder, "*MTL.json"))

    assert len(fp_mtl) == 1, f"Multiple MTL-files found with '{os.path.join(ls_folder, '*MTL.json')}'. {fp_mtl}"

    with open(fp_mtl[0], "r") as json_mtl:
        mtl = json.load(json_mtl)['LANDSAT_METADATA_FILE']

    assert mtl["PRODUCT_CONTENTS"]["PROCESSING_LEVEL"] == "L2SP"
    
    return mtl

def untar(file, folder):
    """Unpack a tar-file.

    Parameters
    ----------
    file : str
        Path to tar-file.
    folder : str
        Path to folder in which the tar-file will be extracted.

    Returns
    -------
    str
        Folder in which the tar-file was extracted.
    """
    target_folder = os.path.join(folder, os.path.splitext(file)[0])

    with tarfile.open(os.path.join(folder, file)) as tar:
        tar.extractall(target_folder)

    return target_folder

def find_file(folder, string, expected_amount = None):
    """Search for files in 'folder' containing 'string' in their filepath. Can
    use "*" as wildcards.

    Parameters
    ----------
    folder : str
        Path to folder to be searched.
    string : str
        String to search in filenames.
    expected_amount : int, optional
        Check if more or less files were found than exepected, by default None

    Returns
    -------
    list
        Paths to the files that contain 'string'.
    """
    fps = glob.glob(os.path.join(folder, string))
    if isinstance(expected_amount, int):
        assert len(fps) == expected_amount, f"{fps}"
    return fps

def find_files(ls_folder, to_open):
    """Given a folder with an unpacked Landsat scene, look for filepaths of 
    bands specified by 'to_open'.

    Parameters
    ----------
    ls_folder : str
        Path to folder with unpacked Landsat scene.
    to_open : list
        List of strings, e.g. ["red", "green"].

    Returns
    -------
    dict
        Keys are the strings from 'to_open', values are matched filepaths.
    """
    search_paths = search_path()
    mtl = open_mtl(ls_folder)
    ls_number = ls_number_from_mtl(mtl)
    fps = {k: find_file(ls_folder, v, 1)[0] for k, v in search_paths[ls_number].items() if k in to_open}
    return fps

def open_as_xr(fp, name):
    """Open a file as a xarray.Dataset and change the default band coordinate
    to 'name'.

    Parameters
    ----------
    fp : str
        Path to GeoTIFF.
    name : str
        Name to be given as coordinate in the 'band' dimension.

    Returns
    -------
    xarray.Dataset
        Dataset with data.
    """
    return xr.open_dataset(fp, chunks = "auto").assign_coords({"band": [name]})

def albedo_weight():
    """Default weights to calculate albedo.

    Returns
    -------
    dict
        Default weights to be applied to the reflectanes in order to calculate
        albedo, for Landsat 7 and 8 Collection-2 Level-2 scenes.
    """
    weights = {
                5: {
                    "blue": 0.116,
                    "green": 0.010,
                    "red": 0.364,
                    "nir": 0.360,
                    "offset": 0.032,
                    },
                7: {
                    "blue": 0.085,
                    "green": 0.057,
                    "red": 0.349,
                    "nir": 0.359,
                    "offset": 0.033,
                    },
                8: {
                    "blue": 0.079,
                    "green": 0.083,
                    "red": 0.334,
                    "nir": 0.360,
                    "offset": 0.031,
                    },
                9: {
                    "blue": 0.079,
                    "green": 0.083,
                    "red": 0.334,
                    "nir": 0.360,
                    "offset": 0.031,
                    },
            }
    return weights

def search_path():
    """Link between general band-names and Landsat specific ones.

    Returns
    -------
    dict
        Keys are general names (e.g. "red", "nir") and values are strings to
        be used with glob.glob.
    """
    search_paths = {7: {
                    "blue":     "*SR_B1.TIF",
                    "green":    "*SR_B2.TIF",
                    "red":      "*SR_B3.TIF",
                    "nir":      "*SR_B4.TIF",
                    "swir1":    "*SR_B5.TIF",
                    "therm":    "*ST_B6.TIF",
                    "swir2":    "*SR_B7.TIF",
                    "pixel_qa": "*QA_PIXEL.TIF",
                    "radsat_qa":"*QA_RADSAT.TIF",
                    "therm_qa": "*ST_QA.TIF",
                    },
                
                5: {
                    "blue":     "*SR_B1.TIF",
                    "green":    "*SR_B2.TIF",
                    "red":      "*SR_B3.TIF",
                    "nir":      "*SR_B4.TIF",
                    "swir1":    "*SR_B5.TIF",
                    "therm":    "*ST_B6.TIF",
                    "swir2":    "*SR_B7.TIF",
                    "pixel_qa": "*QA_PIXEL.TIF",
                    "radsat_qa":"*QA_RADSAT.TIF",
                    "therm_qa": "*ST_QA.TIF",
                    },

                8: {
                    "blue":     "*SR_B2.TIF",
                    "green":    "*SR_B3.TIF",
                    "red":      "*SR_B4.TIF",
                    "nir":      "*SR_B5.TIF",
                    "swir1":    "*SR_B6.TIF",
                    "swir2":    "*SR_B7.TIF",
                    "therm":    "*ST_B10.TIF",
                    "pixel_qa": "*QA_PIXEL.TIF",
                    "radsat_qa":"*QA_RADSAT.TIF",
                    "therm_qa": "*ST_QA.TIF",
                    },
                9: {
                    "blue":     "*SR_B2.TIF",
                    "green":    "*SR_B3.TIF",
                    "red":      "*SR_B4.TIF",
                    "nir":      "*SR_B5.TIF",
                    "swir1":    "*SR_B6.TIF",
                    "swir2":    "*SR_B7.TIF",
                    "therm":    "*ST_B10.TIF",
                    "pixel_qa": "*QA_PIXEL.TIF",
                    "radsat_qa":"*QA_RADSAT.TIF",
                    "therm_qa": "*ST_QA.TIF",
                    },
                }
    return search_paths

def valid_range():
    """Valid DN ranges for Landsat 7 and 8 Collection-2 Level-2 bands.

    Returns
    -------
    dict
        Keys are general names (e.g. "red", "nir") and values are tuples with
        valid DN ranges.

    Notes
    -----
    These values are not stored in the MTL-file unfortunately and come from 
    the product-guides:
    https://www.usgs.gov/media/files/landsat-4-7-collection-2-level-2-science-product-guide
    https://www.usgs.gov/media/files/landsat-8-collection-2-level-2-science-product-guide 

    """


    valid_sr_range = {
        7: {
            "blue":     (7273, 43636),
            "green":    (7273, 43636),
            "red":      (7273, 43636),
            "nir":      (7273, 43636),
            "swir1":    (7273, 43636),
            "swir2":    (7273, 43636), # Assuming, not in table...
            "therm":    (1, 65535),
            "therm_qa":  (0, 32767),
            },
        8: {
            "aer":      (7273, 43636),
            "blue":     (7273, 43636),
            "green":    (7273, 43636),
            "red":      (7273, 43636),
            "nir":      (7273, 43636),
            "swir1":    (1, 65535),
            "swir2":    (7273, 43636),
            "therm":    (1, 65535),
            "therm_qa": (0, 32767),
            },
        }
    valid_sr_range[4] = valid_sr_range[7]
    valid_sr_range[5] = valid_sr_range[7]
    valid_sr_range[9] = valid_sr_range[8]

    return valid_sr_range

def mask_invalid(ds, valid_range):
    """Mask data that is outside of a valid range.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to be clipped.
    valid_range : dict
        Keys are general band-names, values are tuples of floats with lower and
        upper boundary.

    Returns
    -------
    xarray.Dataset
        Datasat in which values that lie outside the valid range are 
        set to nan.
    """

    min_xr = xr.DataArray(data = np.array(list(valid_range.values()))[:,0], 
                        coords = {"band": list(valid_range.keys())})
    max_xr = xr.DataArray(data = np.array(list(valid_range.values()))[:,1], 
                        coords = {"band": list(valid_range.keys())})

    return ds.where((ds.band_data >= min_xr) & (ds.band_data <= max_xr))

def check_projs_geots(files, ref_proj_geot = None):
    """Check if different files all have the same projection and geotransform.

    Parameters
    ----------
    files : list
        Paths to GeoTIFFS to be checked.
    ref_proj_geot : tuple, optional
        Extra projection and geotransform to check the files against. Default 
        is None.

    Returns
    -------
    str
        The uniform projection across all the files or the first projection in 
        case the projections are not uniform among the files.
    tuple
        The uniform geotransform across all the files or the first geotransform in 
        case the projections are not uniform among the files.    
    """
    projs_geots = np.array([proj_geot(fp) for fp in files], dtype=object)
    consistent_projs = np.unique(projs_geots[:,0]).size == 1     
    consistent_geots = np.unique(projs_geots[:,1]).size == 1
    assert consistent_projs
    assert consistent_geots
    proj = projs_geots[:,0][0]
    geot = projs_geots[:,1][0]
    if not isinstance(ref_proj_geot, type(None)):
        assert proj == ref_proj_geot[0]
        assert geot == ref_proj_geot[1]
    return proj, geot

def dl_landsat_test(folder):
    out_file = os.path.join(folder, "LE07_L2SP_177040_20210707_20210802_02_T1.tar")
    url = 'https://storage.googleapis.com/fao-cog-data/LE07_L2SP_177040_20210707_20210802_02_T1.tar'
    download_url(url, out_file, None)
    return out_file

if __name__ == "__main__":

    max_lst_uncertainty = 1.0

    folder = r"/Users/hmcoerver/On My Mac/ndvi_r0_test"

    latlim = [28.9, 29.7]
    lonlim = [30.2, 31.2]

    left = lonlim[0]
    bottom = latlim[0]
    right = lonlim[1]
    top = latlim[1]

    final_bb = [left, bottom, right, top]

    # ds = main(folder, max_lst_uncertainty = max_lst_uncertainty, final_bb = final_bb)

# #%%

# import xarray as xr
# from pywapor.general.processing_functions import save_ds, open_ds
# import rasterio


# latlim = [28.9, 29.7]
# lonlim = [30.2, 31.2]
# left = lonlim[0]
# bottom = latlim[0]
# right = lonlim[1]
# top = latlim[1]

# final_bb = [left, bottom, right, top]

# def transform_bb(src_crs, dst_crs, bb):
#     bb =rasterio.warp.transform_bounds(src_crs, dst_crs, *bb, densify_pts=21)
#     return bb

# ds = open_ds(r"/Users/hmcoerver/On My Mac/ndvi_r0_test/test_in.nc")

# target_crs = rasterio.crs.CRS.from_epsg(4326)

# bb = transform_bb(target_crs, ds.rio.crs, final_bb)

# ds = ds.rio.clip_box(*bb)
# ds = ds.rio.pad_box(*bb)

# ds = ds.rio.reproject(target_crs)

