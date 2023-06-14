"""Functions that convert different landuse classifications into the ETLook variables
`land_mask`, `lue_max`, `rs_min` and `z_obst_max`.
"""
import numpy as np
from pywapor.general.logger import log

def lulc_to_x(ds, var, convertor, in_var = None, out_var = None):
    """Convert values in a classified map to other values based on a dictionary.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing `var`.
    var : str
        Variable name.
    convertor : dict
        Dictionary with keys corresponding to values in `var` and values the replacements.
    in_var : str, optional
        Overwrites `var`, by default None.
    out_var : str, optional
        New variable name in case `var` should not be overwritten, by default None.

    Returns
    -------
    xr.Dataset
        Dataset with the replaced values.
    """

    log.info(f"--> Calculating `{out_var}` from `lulc`.")

    if not isinstance(in_var, type(None)):
        var = in_var

    new_data = ds[var] * np.nan

    for class_number in np.unique(ds[var]):
        if class_number in convertor.keys():
            new_data = new_data.where(ds[var] != class_number, 
                                        convertor[class_number])

    if "sources" in ds[var].attrs.keys():
        new_data.attrs = {"sources": ds[var].attrs["sources"]}

    if not isinstance(out_var, type(None)):
        ds[out_var] = new_data
    else: 
        ds[var] = new_data

    return ds

def wapor_to_land_mask():
    """Generate a converter dictionary to be used by `lulc.lulc_to_x` to convert
    from WaPOR-lulc to the ETlook `land_mask`.

    Returns
    -------
    dict
        Describes the conversions to be made.
    """

    convertor = {
                    20: 1,
                    30: 1,
                    42: 1,
                    43: 1,
                    50: 3,
                    60: 1,
                    80: 2,
                    90: 1,
                    116: 1,
                    126 : 1,
                    33 : 1,
                    41 : 1,
                }

    return convertor

def wapor_to_lue_max():
    """Generate a converter dictionary to be used by `lulc.lulc_to_x` to convert
    from WaPOR-lulc to the ETlook `lue_max`.

    Returns
    -------
    dict
        Describes the conversions to be made.
    """

    convertor = {
                    20:	2.10,	# Shrubland
                    30:	2.39,	# Grassland
                    41:	2.7,	# Cropland, rainfed
                    42:	2.7,	# Cropland, irrigated or under water management
                    43:	2.7,	# Cropland, fallow
                    50:	1.42,	# Built-up
                    60:	1.42,	# Bare / sparse vegetation
                    70:	0.0,	# Permament snow / ice
                    80:	0.0,	# Water bodies
                    81:	1.43,	# Temporary water bodies
                    90:	2.1,	# Shrub or herbaceous cover, flooded
                    111: 1.98,	# Tree cover: closed, evergreen needle-leaved
                    112: 2.56,	# Tree cover: closed, evergreen broadleaved
                    114: 1.99,	# Tree cover: closed, deciduous broadleaved
                    115: 2.23,	# Tree cover: closed, mixed type
                    116: 2.23,	# Tree cover: closed, unknown type
                    121: 1.98,	# Tree cover: open, evergreen needle-leaved
                    122: 2.56,	# Tree cover: open, evergreen broadleaved
                    123: 1.57,	# Tree cover: open, deciduous needle-leaved
                    124: 2.48,	# Tree cover: open, deciduous broadleaved
                    125: 2.23,	# Tree cover: open, mixed type
                    126: 2.23,	# Tree cover: open, unknown type
                    200: 0.0,	# Sea water
    }

    return convertor

def wapor_to_rs_min():
    """Generate a converter dictionary to be used by `lulc.lulc_to_x` to convert
    from WaPOR-lulc to the ETlook `rs_min`.

    Returns
    -------
    dict
        Describes the conversions to be made.
    """

    convertor = {
                20 : 175,
                30 : 175,
                42 : 125,
                43 : 125,
                50 : 400,
                60 : 100,
                80 : 100,
                90 : 150,
                116 : 180,
                126 : 250,
                33 : 175,
                41 : 150,
    }

    return convertor

def wapor_to_z_obst_max():
    """Generate a converter dictionary to be used by `lulc.lulc_to_x` to convert
    from WaPOR-lulc to the ETlook `z_obst_max`.

    Returns
    -------
    dict
        Describes the conversions to be made.
    """

    convertor = {
                    20 : 1.0,
                    30 : 2.0,
                    42 : 1.5,
                    43 : 1.5,
                    50 : 10.0,
                    60 : 0.1,
                    80 : 0.1,
                    90 : 2.0,
                    116 : 5.0,
                    126 : 3.0,
                    33 : 8.0,
                    41 : 2.0,
    }

    return convertor

def globcover_to_land_mask():
    """Generate a converter dictionary to be used by `lulc.lulc_to_x` to convert
    from GLOBCOVER-lulc to the ETlook `land_mask`.

    Returns
    -------
    dict
        Describes the conversions to be made.
    """

    convertor = {
                    11: 1,    
                    14:	1,    
                    20:	1,    
                    30:	1,    
                    40:	1,    
                    50:	1,    
                    60:	1,    
                    70:	1,    
                    90:	1,    
                    100: 1,   	 
                    110: 1,   	 
                    120: 1,   	 
                    130: 1,  	 
                    140: 1,  	 
                    150: 1,  	 
                    160: 1,  	 
                    170: 1,  	 
                    180: 1,  	 
                    190: 3,  	 
                    200: 1,  	 
                    210: 2,  	 
                    220: 1,  	 
                    230: 0  	 
        }

    return convertor

def globcover_to_lue_max():
    """Generate a converter dictionary to be used by `lulc.lulc_to_x` to convert
    from GLOBCOVER-lulc to the ETlook `lue_max`.

    Returns
    -------
    dict
        Describes the conversions to be made.
    """

    convertor = {
                11: 3.04,    
                14:	3.04,    
                20:	3.04,    
                30:	3.04,    
                40:	2.45,    
                50:	2.66,    
                60:	2.66,    
                70:	2.5,    
                90:	2.5,    
                100: 2.54,   	 
                110: 2.54,   	 
                120: 2.54,   	 
                130: 2.54,  	 
                140: 2.3,  	 
                150: 1.57,  	 
                160: 2.66,  	 
                170: 1.57,  	 
                180: 1.57,  	 
                190: 1.57,  	 
                200: 1.57,  	 
                210: 0,  	 
                220: 0,  	 
                230: 0  	 
        }

    return convertor

def globcover_to_rs_min():
    """Generate a converter dictionary to be used by `lulc.lulc_to_x` to convert
    from GLOBCOVER-lulc to the ETlook `rs_min`.

    Returns
    -------
    dict
        Describes the conversions to be made.
    """

    convertor = {
                11: 200,    
                14:	200,    
                20:	150,    
                30:	150,    
                40:	100,    
                50:	120,    
                60:	100,    
                70:	150,    
                90:	180,    
                100: 175,   	 
                110: 150,   	 
                120: 350,   	 
                130: 175,  	 
                140: 250,  	 
                150: 150,  	 
                160: 250,  	 
                170: 200,  	 
                180: 300,  	 
                190: 100,  	 
                200: 100,  	 
                210: 100,  	 
                220: 100,  	 
                230: 0  	 
        }

    return convertor

def globcover_to_z_obst_max():
    """Generate a converter dictionary to be used by `lulc.lulc_to_x` to convert
    from GLOBCOVER-lulc to the ETlook `z_obst_max`.

    Returns
    -------
    dict
        Describes the conversions to be made.
    """

    convertor = {
            11: 4.0,    
            14:	4.0,    
            20:	2.0,    
            30:	3.5,    
            40:	0.1,    
            50:	0.6,    
            60:	1.2,    
            70:	2.0,    
            90:	5.0,    
            100: 8.0,   	 
            110: 2.0,   	 
            120: 8.0,   	 
            130: 4.0,  	 
            140: 2.0,  	 
            150: 1.0,  	 
            160: 0.3,  	 
            170: 6.0,  	 
            180: 10,  	 
            190: 10,
            200: 0.1,
            210: 0.1,
            220: 0.1, 
            230: 0,
    }

    return convertor
