import copy
import pywapor.se_root as se_root
from functools import partial
import types
import numpy as np
from pywapor.general.logger import log
from pywapor.enhancers.dms.thermal_sharpener import sharpen

def find_setting(sources, setting_str, max_length = np.inf, min_length = 0):
    """Search a `sources` dictionary for a specific key and return a list of `source.product`s that
    include that key.

    Parameters
    ----------
    sources : dict
        Source configuration for `pre_et_look` and `pre_se_root`.
    setting_str : str
        Key to search for.
    max_length : int, optional
        Give a warning of more than `max_length` products have been found, by default np.inf.
    min_length : int, optional
        Give a warning if less than `min_length` products have been found, by default 0.

    Returns
    -------
    list
        List of `source.product`s that contain `setting_str` in their nested dictionary.
    """
    example_sources = list()
    for var, x in sources.items():
        x_ = x.get("products", [])
        prod = [product for product in x_ if setting_str in product.keys()]
        if len(prod) >= 1:
            for pro in prod:
                if pro[setting_str]:
                    example_source = (pro["source"], pro["product_name"])
                    if isinstance(example_source[0], types.FunctionType):
                        example_source = (example_source[0].__name__, example_source[1])
                    elif isinstance(example_source[0], partial):
                        example_source = (example_source[0].func.__name__, example_source[1])
                    if example_source not in example_sources:
                        example_sources.append(example_source)
        else:
            if setting_str in x.keys():
                example_sources.append(var)
    if len(example_sources) > max_length:
            log.warning(f"--> Found more than {max_length} products for `{setting_str}`.")
    if len(example_sources) > max_length or len(example_sources) < min_length:
        log.warning(f"--> Didn't find any products for `{setting_str}`.")
    return example_sources

def pre_et_look_levels(level = "level_1", bin_length = "DEKAD"):
    """Create a default `pre_et_look` `sources` dictionary.

    Parameters
    ----------
    level : "level_1" | "level_2" | "level_3" | "level_2_v3"
        For which level to create the `sources`, by default "level_1".
    bin_length : str, optional
        Defines the bin length with which the `sources` will be used, by default "DEKAD".

    Returns
    -------
    dict
        Dictionary with variable names as keys and dictionaries as values.
    """

    se_root_dler = partial(se_root.se_root, bin_length = bin_length, 
                            sources = level)

    level_1 = {

        "ndvi": {
            "products": [
                {
                    "source": "MODIS",
                    "product_name": "MOD13Q1.061",
                    "enhancers": "default",
                    "is_example": True
                },
                {
                    "source": "MODIS",
                    "product_name": "MYD13Q1.061",
                    "enhancers": "default",
                }
            ],
            "composite_type": "mean",
            "temporal_interp": "linear",
            "spatial_interp": "nearest",
            },

        "r0": {
            "products": [
                {
                    "source": "MODIS",
                    "product_name": "MCD43A3.061",
                    "enhancers": "default",
                },
            ],
            "composite_type": "mean",
            "temporal_interp": "linear",
            "spatial_interp": "bilinear",
            },

        "se_root": {
            "products": [
                {
                    "source": se_root_dler,
                    "product_name": "v2",
                    "enhancers": "default",
                },
            ],
            "composite_type": "max",
            "temporal_interp": None,
            "spatial_interp": "bilinear",
        },

        "p": {
            "products": [
                {
                    "source": "CHIRPS",
                    "product_name": "P05",
                    "enhancers": "default",
                },
            ],
            "composite_type": "mean",
            "temporal_interp": "linear",
            "spatial_interp": "bilinear",
            },

        "z": {
            "products": [
                {
                    "source": "SRTM",
                    "product_name": "30M",
                    "enhancers": "default",
                },
            ],
            "composite_type": None,
            "temporal_interp": None,
            "spatial_interp": "bilinear",
            },

        "ra": {
            "products": [
                {
                    "source": "MERRA2",
                    "product_name": "M2T1NXRAD.5.12.4",
                    "enhancers": "default",
                },
            ],
            "composite_type": "mean",
            "temporal_interp": "linear",
            "spatial_interp": "bilinear",
            },

        "t_air": {
            "products": [
                {
                    "source": "GEOS5",
                    "product_name": "inst3_2d_asm_Nx",
                    "enhancers": "default",
                },
            ],
            "composite_type": "mean",
            "temporal_interp": "linear",
            "spatial_interp": "bilinear",
            },

        "t_air_max": {
            "products": [
                {
                    "source": "GEOS5",
                    "product_name": "inst3_2d_asm_Nx",
                    "enhancers": "default",
                },
            ],
            "composite_type": "max",
            "temporal_interp": "linear",
            "spatial_interp": "bilinear",
            },

        "t_air_min": {
            "products": [
                {
                    "source": "GEOS5",
                    "product_name": "inst3_2d_asm_Nx",
                    "enhancers": "default",
                },
            ],
            "composite_type": "min",
            "temporal_interp": "linear",
            "spatial_interp": "bilinear",
            },

        "u2m": {
            "products": [
                {
                    "source": "GEOS5",
                    "product_name": "inst3_2d_asm_Nx",
                    "enhancers": "default",
                },
            ],
            "composite_type": "mean",
            "temporal_interp": "linear",
            "spatial_interp": "bilinear",
            },

        "v2m": {
            "products": [
                {
                    "source": "GEOS5",
                    "product_name": "inst3_2d_asm_Nx",
                    "enhancers": "default",
                },
            ],
            "composite_type": "mean",
            "temporal_interp": "linear",
            "spatial_interp": "bilinear",
            },

        "qv": {
            "products": [
                {
                    "source": "GEOS5",
                    "product_name": "inst3_2d_asm_Nx",
                    "enhancers": "default",
                },
            ],
            "composite_type": "mean",
            "temporal_interp": "linear",
            "spatial_interp": "bilinear",
            },

        "p_air": {
            "products": [
                {
                    "source": "GEOS5",
                    "product_name": "inst3_2d_asm_Nx",
                    "enhancers": "default",
                },
            ],
            "composite_type": "mean",
            "temporal_interp": "linear",
            "spatial_interp": "bilinear",
            },

        "p_air_0": {
            "products": [
                {
                    "source": "GEOS5",
                    "product_name": "inst3_2d_asm_Nx",
                    "enhancers": "default",
                },
            ],
            "composite_type": "mean",
            "temporal_interp": "linear",
            "spatial_interp": "bilinear",
            },

        "wv": {
            "products": [
                {
                    "source": "GEOS5",
                    "product_name": "inst3_2d_asm_Nx",
                    "enhancers": "default",
                },
            ],
            "composite_type": "mean",
            "temporal_interp": "linear",
            "spatial_interp": "bilinear",
            },

        "land_mask": {
            "products": [
                {
                    "source": "GLOBCOVER",
                    "product_name": "2009_V2.3_Global",
                    "enhancers": "default",
                },
            ],
            "composite_type": None,
            "temporal_interp": None,
            "spatial_interp": "nearest",
            },

        "rs_min": {
            "products": [
                {
                    "source": "GLOBCOVER",
                    "product_name": "2009_V2.3_Global",
                    "enhancers": "default",
                },
            ],
            "composite_type": None,
            "temporal_interp": None,
            "spatial_interp": "nearest",
            },

        "z_obst_max": {
            "products": [
                {
                    "source": "GLOBCOVER",
                    "product_name": "2009_V2.3_Global",
                    "enhancers": "default",
                },
            ],
            "composite_type": None,
            "temporal_interp": None,
            "spatial_interp": "nearest",
            },

    }

    statics = [
                'lw_offset', 'lw_slope', 'z_oro', 'rn_offset', 'rn_slope', 't_amp_year', 't_opt', 'vpd_slope',
                # 'land_mask', 'rs_min', 'z_obst_max' # NOTE generated from lulc
            ]

    for var in statics:

        level_1[var] =  {
            "products": [
                {
                    "source": "STATICS",
                    "product_name": "WaPOR2",
                    "enhancers": "default",
                },
            ],
            "composite_type": None,
            "temporal_interp": None,
            "spatial_interp": "bilinear",
        }

    level_2 = copy.deepcopy(level_1)

    level_2["ndvi"] = {

            "products": [
                {
                    "source": "PROBAV",
                    "product_name": "S5_TOC_100_m_C1",
                    "enhancers": "default",
                    "is_example": True
                },
            ],
            "composite_type": "mean",
            "temporal_interp": "linear",
            "spatial_interp": "nearest",
            }

    level_2["r0"] = {

            "products": [
                {
                    "source": "PROBAV",
                    "product_name": "S5_TOC_100_m_C1",
                    "enhancers": "default",
                },
            ],
            "composite_type": "mean",
            "temporal_interp": "linear",
            "spatial_interp": "nearest",
            }

    level_3 = copy.deepcopy(level_1)

    level_3["ndvi"] = {

            "products": [
                {
                    "source": "LANDSAT",
                    "product_name": "LT05_SR",
                    "enhancers": "default",
                    "is_example": True
                },
                {
                    "source": "LANDSAT",
                    "product_name": "LE07_SR",
                    "enhancers": "default",
                    "is_example": True
                },
                {
                    "source": "LANDSAT",
                    "product_name": "LC08_SR",
                    "enhancers": "default",
                    "is_example": True
                },
                {
                    "source": "LANDSAT",
                    "product_name": "LC09_SR",
                    "enhancers": "default",
                    "is_example": True
                },
            ],
            "composite_type": "mean",
            "temporal_interp": "linear",
            "spatial_interp": "nearest",
            }

    level_3["r0"] = {

            "products": [
                {
                    "source": "LANDSAT",
                    "product_name": "LT05_SR",
                    "enhancers": "default",
                },
                {
                    "source": "LANDSAT",
                    "product_name": "LE07_SR",
                    "enhancers": "default",
                },
                {
                    "source": "LANDSAT",
                    "product_name": "LC08_SR",
                    "enhancers": "default",
                },
                {
                    "source": "LANDSAT",
                    "product_name": "LC09_SR",
                    "enhancers": "default",
                },
            ],
            "composite_type": "mean",
            "temporal_interp": "linear",
            "spatial_interp": "nearest",
            }

    level_2_v3 = dict()

    level_2_v3["ndvi"] = {

            "products": [
                {
                    "source": "SENTINEL2",
                    "product_name": "S2MSI2A_R60m",
                    "enhancers": "default",
                    "is_example": True
                },
            ],
            "composite_type": "mean",
            "temporal_interp":  {
                                "method": "whittaker",
                                # "plot_folder": r"",
                                "valid_drange": [-1.0, 1.0],
                                "max_dist": np.timedelta64(15, "D"),
                                "lambdas": 100.,
                    },
            "spatial_interp": "nearest",
            }

    level_2_v3["r0"] = {

            "products": [
                {
                    "source": "SENTINEL2",
                    "product_name": "S2MSI2A_R60m",
                    "enhancers": "default",
                },
            ],
            "composite_type": "mean",
            "temporal_interp":  {
                                "method": "whittaker",
                                # "plot_folder": r"",
                                "valid_drange": [0.0, 1.0],
                                "max_dist": np.timedelta64(15, "D"),
                                "lambdas": 100.,
                    },
            "spatial_interp": "nearest",
            }

    level_2_v3["se_root"] = {
            "products": [
                {
                    "source": se_root_dler,
                    "product_name": "v3",
                    "enhancers": "default",
                },
            ],
            "composite_type": "max",
            "temporal_interp": None,
            "spatial_interp": "bilinear",
        }

    level_2_v3["p"] = {
        "products": [
            {
                "source": "CHIRPS",
                "product_name": "P05",
                "enhancers": "default",
            },
        ],
        "composite_type": "mean",
        "temporal_interp": "linear",
        "spatial_interp": "bilinear",
        }

    level_2_v3["z"] = {
        "products": [
            {
                "source": "COPERNICUS",
                "product_name": "GLO90",
                "enhancers": "default",
            },
        ],
        "composite_type": None,
        "temporal_interp": None,
        "spatial_interp": "bilinear",
        }

    for var, composite_type in [("t_air", "mean"), ("t_air_min", "min"), ("t_air_max", "max"), 
                                # ("t_dew", "mean"), # NOTE ETLook usees t_dew to calc `vp`, but that one is directly available from agERA5.
                                # ("rh", "mean"), 
                                ("u", "mean"), ("vp", "mean"), ("ra", "mean")]:
        level_2_v3[var] = {
            "products": [
                {
                    "source": "ERA5",
                    "product_name": "sis-agrometeorological-indicators",
                    "enhancers": "default",
                },
            ],
            "composite_type": composite_type,
            "temporal_interp": "linear",
            "spatial_interp": "bilinear",
            }

    for var in ["p_air", "p_air_0"]:
        level_2_v3[var] = {
            "products": [
                {
                    "source": "ERA5",
                    "product_name": 'reanalysis-era5-single-levels',
                    "enhancers": "default",
                },
            ],
            "composite_type": "mean",
            "temporal_interp": "linear",
            "spatial_interp": "bilinear",
            }

    statics_v2 = [
                'z_oro', 'rn_offset', 'rn_slope', 't_amp_year', 't_opt', 'vpd_slope',
                'land_mask', 'rs_min', 'z_obst_max'
            ]

    for var in statics_v2:
        level_2_v3[var] = {
            'products': [
                {
                    'source': "STATICS",
                    'product_name': "WaPOR2",
                    'enhancers': 'default'
                },
            ],
        'composite_type': None,
        'temporal_interp': None,
        'spatial_interp': 'bilinear'}

    statics_v3 = [
                'lw_offset', 'lw_slope'
            ]

    for var in statics_v3:
        level_2_v3[var] = {
            'products': [
                {
                    'source': "STATICS",
                    'product_name': "WaPOR3",
                    'enhancers': 'default'
                },
            ],
        'composite_type': None,
        'temporal_interp': None,
        'spatial_interp': 'bilinear'}
    



    levels = {
            "level_1": level_1,
            "level_2": level_2,
            "level_3": level_3,

            "level_2_v3": level_2_v3
                }

    return levels[level]

def pre_se_root_levels(level = "level_1"):
    """Create a default `pre_se_root` `sources` dictionary.

    Parameters
    ----------
    level : "level_1" | "level_2" | "level_3" | "level_2_v3"
        For which level to create the `sources`, by default "level_1".

    Returns
    -------
    dict
        Dictionary with variable names as keys and dictionaries as values.
    """

    level_1 = {

        "ndvi": {
            "products": [
                {
                    "source": "MODIS",
                    "product_name": "MOD13Q1.061",
                    "enhancers": "default",
                    "is_example": True,
                },
                {
                    "source": "MODIS",
                    "product_name": "MYD13Q1.061",
                    "enhancers": "default", 
                }
            ],
            "temporal_interp": "linear",
            "spatial_interp": "nearest",
            },

        "lst": {
            "products": [
                {
                    "source": "MODIS",
                    "product_name": "MOD11A1.061",
                    "enhancers": "default",
                },
                {
                    "source": "MODIS",
                    "product_name": "MYD11A1.061",
                    "enhancers": "default",
                },
            ],
            "temporal_interp": None,
            "spatial_interp": "nearest",
        },

        "t_air": {
            "products": [
                {
                    "source": "GEOS5",
                    "product_name": "inst3_2d_asm_Nx",
                    "enhancers": "default",
                },
            ],
            "temporal_interp": "linear",
            "spatial_interp": "bilinear",
            },

        "t_air_max": {
            "products": [
                {
                    "source": "GEOS5",
                    "product_name": "inst3_2d_asm_Nx",
                    "enhancers": "default",
                },
            ],
            "temporal_interp": "linear",
            "spatial_interp": "bilinear",
            },

        "t_air_min": {
            "products": [
                {
                    "source": "GEOS5",
                    "product_name": "inst3_2d_asm_Nx",
                    "enhancers": "default",
                },
            ],
            "temporal_interp": "linear",
            "spatial_interp": "bilinear",
            },

        "u2m": {
            "products": [
                {
                    "source": "GEOS5",
                    "product_name": "inst3_2d_asm_Nx",
                    "enhancers": "default",
                },
            ],
            "temporal_interp": "linear",
            "spatial_interp": "bilinear",
            },

        "v2m": {
            "products": [
                {
                    "source": "GEOS5",
                    "product_name": "inst3_2d_asm_Nx",
                    "enhancers": "default",
                },
            ],
            "temporal_interp": "linear",
            "spatial_interp": "bilinear",
            },

        "qv": {
            "products": [
                {
                    "source": "GEOS5",
                    "product_name": "inst3_2d_asm_Nx",
                    "enhancers": "default",
                },
            ],
            "temporal_interp": "linear",
            "spatial_interp": "bilinear",
            },

        "wv": {
            "products": [
                {
                    "source": "GEOS5",
                    "product_name": "inst3_2d_asm_Nx",
                    "enhancers": "default",
                },
            ],
            "temporal_interp": "linear",
            "spatial_interp": "bilinear",
            },

        "p_air": {
            "products": [
                {
                    "source": "GEOS5",
                    "product_name": "inst3_2d_asm_Nx",
                    "enhancers": "default",
                },
            ],
            "temporal_interp": "linear",
            "spatial_interp": "bilinear",
            },

        "p_air_0": {
            "products": [
                {
                    "source": "GEOS5",
                    "product_name": "inst3_2d_asm_Nx",
                    "enhancers": "default",
                },
            ],
            "temporal_interp": "linear",
            "spatial_interp": "bilinear",
            },
    }

    statics = ["r0_bare", "r0_full"]

    for var in statics:

        level_1[var] =  {
            "products": [
                {
                    "source": "STATICS",
                    "product_name": "WaPOR2",
                    "enhancers": "default",
                },
            ],
            "temporal_interp": None,
            "spatial_interp": "bilinear",
        }

    level_2 = copy.deepcopy(level_1)

    level_2["ndvi"] = {

            "products": [
                {
                    "source": "PROBAV",
                    "product_name": "S5_TOC_100_m_C1",
                    "enhancers": "default",
                    "is_example": True
                },
            ],
            "composite_type": "mean",
            "temporal_interp": "linear",
            "spatial_interp": "nearest",
            }

    level_3 = copy.deepcopy(level_1)

    level_3["ndvi"] = {

            "products": [
                {
                    "source": "LANDSAT",
                    "product_name": "LT05_SR",
                    "enhancers": "default",
                    "is_example": True
                },
                {
                    "source": "LANDSAT",
                    "product_name": "LE07_SR",
                    "enhancers": "default",
                    "is_example": True
                },
                {
                    "source": "LANDSAT",
                    "product_name": "LC08_SR",
                    "enhancers": "default",
                    "is_example": True
                },
                {
                    "source": "LANDSAT",
                    "product_name": "LC09_SR",
                    "enhancers": "default",
                    "is_example": True
                },
            ],
            "composite_type": "mean",
            "temporal_interp": "linear",
            "spatial_interp": "nearest",
            }

    level_3["lst"] = {
            "products": [
                {
                    "source": "LANDSAT",
                    "product_name": "LT05_ST",
                    "enhancers": "default",
                },
                {
                    "source": "LANDSAT",
                    "product_name": "LE07_ST",
                    "enhancers": "default",
                },
                {
                    "source": "LANDSAT",
                    "product_name": "LC08_ST",
                    "enhancers": "default",
                },
                {
                    "source": "LANDSAT",
                    "product_name": "LC09_ST",
                    "enhancers": "default",
                },
            ],
            "temporal_interp": None,
            "spatial_interp": "nearest",
        }

    level_2_v3 = dict()

    level_2_v3["ndvi"] = {
        "products": [
            {
                "source": "SENTINEL2",
                "product_name": "S2MSI2A_R60m",
                "enhancers": "default",
                "is_example": True
            },
        ],
        "temporal_interp":  {
                    "method": "whittaker",
                    # "plot_folder": r"",
                    "valid_drange": [-1.0, 1.0],
                    "max_dist": np.timedelta64(15, "D"),
                    "lambdas": 100.,
        },
        "spatial_interp": "nearest"}

    for var in ["mndwi", "psri", "vari_red_edge", "bsi", "nmdi", "green", "nir"]:
        level_2_v3[var] = {
            'products': [{
                'source': 'SENTINEL2',
                'product_name': 'S2MSI2A_R60m',
                'enhancers': 'default',
            }],
        'temporal_interp': 'linear',
        'spatial_interp': 'nearest'}

    level_2_v3['bt'] = {
        'products': [
            {
                'source': 'VIIRSL1',
                'product_name': 'VNP02IMG',
                'enhancers': 'default'
            },
        ],
        'variable_enhancers': [sharpen],
        "temporal_interp":  {
                    "method": "whittaker",
                    # "plot_folder": r"",
                    "a": 0.95,
                    # "valid_drange": [-1.0, 1.0],
                    "max_dist": np.timedelta64(15, "D"),
                    "lambdas": 100.,
        },
        'spatial_interp': 'nearest'}

    for var in ["u", "t_dew", "p_air_0", "p_air", "t_air", "wv"]:
        level_2_v3[var] = {
            'products': [
                {
                    'source': 'ERA5',
                    'product_name': 'reanalysis-era5-single-levels',
                    'enhancers': 'default'
                },
            ],
        'temporal_interp': "linear",
        'spatial_interp': 'bilinear'}

    for var in ["z", "slope", "aspect"]:
        level_2_v3[var] = {
            'products': [
                {
                    'source': 'COPERNICUS',
                    'product_name': 'GLO90',
                    'enhancers': 'default'
                },
            ],
        'temporal_interp': None,
        'spatial_interp': 'bilinear'}

    for var in ["r0_bare", "r0_full"]:
        level_2_v3[var] = {
            'products': [
                {
                    'source': "STATICS",
                    'product_name': "WaPOR2",
                    'enhancers': 'default'
                },
            ],
        'temporal_interp': "linear",
        'spatial_interp': 'bilinear'}

    levels = {
                "level_1": level_1,
                "level_2": level_2,
                "level_3": level_3,
                "level_2_v3": level_2_v3,
                }

    return levels[level]

if __name__ == "__main__":

    et_look_sources_lvl1 = pre_et_look_levels(level = "level_1", bin_length = "DEKAD")
    et_look_sources_lvl2 = pre_et_look_levels(level = "level_2", bin_length = "DEKAD")
    et_look_sources_lvl3 = pre_et_look_levels(level = "level_3", bin_length = "DEKAD")
    et_look_sources_lvl2_v3 = pre_et_look_levels(level = "level_2_v3", bin_length = "DEKAD")

    se_root_sources_lvl1 = pre_se_root_levels(level = "level_1")
    se_root_sources_lvl2 = pre_se_root_levels(level = "level_2")
    se_root_sources_lvl3 = pre_se_root_levels(level = "level_3")
    se_root_sources_lvl2_v3 = pre_se_root_levels(level = "level_2_v3")

    