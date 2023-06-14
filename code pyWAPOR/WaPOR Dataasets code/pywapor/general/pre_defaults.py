"""Module containing functions that generate some default settings for `pywapor.pre_et_look`
and `pywapor.pre_se_root`.
"""

def constants_defaults():
    """Return a dictionary with default parameters.

    Returns
    -------
    dict
        Default constants.
    """

    c = {

        # et_look
        'nd_min': 0.125,
        'nd_max': 0.8,
        'vc_pow': 0.7,
        'vc_min': 0,
        'vc_max': 0.9677324224821418,
        'lai_pow': -0.45,
        'diffusion_slope': -1.33,
        'diffusion_intercept': 1.15,
        't_min': 0,
        't_max': 50,
        'vp_slope': 0.14,
        'vp_offset': 0.34,
        'int_max': 0.2,
        'tenacity': 1.5,
        'rcan_max': 1000000,
        'ndvi_obs_min': 0.25,
        'ndvi_obs_max': 0.75,
        'obs_fr': 0.25,
        'z_obs': 2,
        'z_b': 100,
        'c1': 1,
        'iter_h': 3,
        'r_soil_pow': -2.1,
        'r_soil_min': 800,
        'se_top': 0.5,
        'porosity': 0.4,
        'r0_grass': 0.23,
        'eps_a': 0.5,

        # se_root
        'z0m_full': 0.1,
        'z0m_bare': 0.001,
        'aod550_i': 0.01,
        'fraction_h_bare': 0.65,
        'fraction_h_full': 0.95,
        'disp_bare': 0.0,
        'disp_full': 0.667,
        'r0_bare_wet': 0.2,
        'IO': 1367.,

        # biomass
        'dh_ap': 52750, 
        'd_s': 704.98, 
        'dh_dp': 211000,
        'ar_slo': 0.0, 
        'ar_int': 0.5,
        'fpar_slope': 1.257, 
        'fpar_offset': -0.161,
        'o2': 20.9,
        'co2_ref': 281,
        'gcgdm': 0.4,
        'phot_eff': 2.49,

        # statics (et_look)
        "t_opt": 25.0,
        "lw_slope": 1.35, 
        "lw_offset": -0.35,
        "rn_slope": 0.92,
        "rn_offset": -61.0,
        "vpd_slope": -0.3,
        "t_amp_year": 8,

        # statics (se_root)
        "r0_bare": 0.38,
        "r0_full": 0.18,

    }

    return c