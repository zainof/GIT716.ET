"""Functions and dictionaries to convert various RS Quality Bitmasks
into numpy boolean arrays.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

def get_pixel_qa_bits(collection, ls_number, level):
    """Returns a dictionary to bridge between labels and bits in
    Landsat pixel_qa bands.

    Parameters
    ----------
    collection : int
        For which collection to get the values.
    ls_number : int
        For which spacecraft to get the values.
    level : int
        For which level to get the values.

    Returns
    -------
    dict
        Keys are labels (different per Landsat mission/product) and values are
        definitions of the bits.

    Examples
    --------
    # Count from right to left.
    # <-etc...9876543210
    #

    Returns True if 2nd bit is set:
    [(0b0000000000000100, False)]

    Returns True if 9th bit is set AND 8th bit is unset.
    [(0b0000001000000000, False)
     (0b0000000100000000, True)]

    Returns True if 11th bit is unset AND 10th bit is unset.
    [(0b0000100000000000, True)
     (0b0000010000000000, True)]
    """


    c2l8lvl1 = {
        'fill':             [(0b0000000000000001, False)],
        'dilated_cloud':    [(0b0000000000000010, False)],
        'cirrus':           [(0b0000000000000100, False)],
        'cloud':            [(0b0000000000001000, False)],
        'cloud_shadow':     [(0b0000000000010000, False)],
        'snow':             [(0b0000000000100000, False)], 
        'clear':            [(0b0000000001000000, False)],
        'water':            [(0b0000000010000000, False)],
        'cloud_nd':         [(0b0000001000000000, True),
                             (0b0000000100000000, True)],
        'cloud_low':        [(0b0000001000000000, True),
                             (0b0000000100000000, False)],
        'cloud_med':        [(0b0000001000000000, False),
                             (0b0000000100000000, True)],
        'cloud_high':       [(0b0000001000000000, False),
                             (0b0000000100000000, False)],
        'cloud_shadow_nd':  [(0b0000100000000000, True),    
                             (0b0000010000000000, True)],   
        'cloud_shadow_low': [(0b0000100000000000, True),    
                             (0b0000010000000000, False)],  
        'cloud_shadow_med': [(0b0000100000000000, False),   
                             (0b0000010000000000, True)],   
        'cloud_shadow_high':[(0b0000100000000000, False),   
                             (0b0000010000000000, False)],  
        'snow_ice_nd':      [(0b0010000000000000, True),    
                             (0b0001000000000000, True)],   
        'snow_ice_low':     [(0b0010000000000000, True),    
                             (0b0001000000000000, False)],  
        'snow_ice_med':     [(0b0010000000000000, False),   
                             (0b0001000000000000, True)],   
        'snow_ice_high':    [(0b0010000000000000, False),   
                             (0b0001000000000000, False)],  
        'cirrus_nd':        [(0b1000000000000000, True),    
                             (0b0100000000000000, True)],   
        'cirrus_low':       [(0b1000000000000000, True),    
                             (0b0100000000000000, False)],  
        'cirrus_med':       [(0b1000000000000000, False),   
                             (0b0100000000000000, True)],   
        'cirrus_high':      [(0b1000000000000000, False),   
                             (0b0100000000000000, False)],  
    }

    c2l7lvl1 = {
        'fill':             [(0b0000000000000001, False)],
        'dilated_cloud':    [(0b0000000000000010, False)],
        'cloud':            [(0b0000000000001000, False)],
        'cloud_shadow':     [(0b0000000000010000, False)],
        'snow':             [(0b0000000000100000, False)], 
        'clear':            [(0b0000000001000000, False)],
        'water':            [(0b0000000010000000, False)],
        'cloud_nd':         [(0b0000001000000000, True),
                             (0b0000000100000000, True)],
        'cloud_low':        [(0b0000001000000000, True),
                             (0b0000000100000000, False)],
        'cloud_med':        [(0b0000001000000000, False),
                             (0b0000000100000000, True)],
        'cloud_high':       [(0b0000001000000000, False),
                             (0b0000000100000000, False)],
        'cloud_shadow_nd':  [(0b0000100000000000, True),    
                             (0b0000010000000000, True)],   
        'cloud_shadow_low': [(0b0000100000000000, True),    
                             (0b0000010000000000, False)],  
        'cloud_shadow_med': [(0b0000100000000000, False),   
                             (0b0000010000000000, True)],   
        'cloud_shadow_high':[(0b0000100000000000, False),   
                             (0b0000010000000000, False)],  
        'snow_ice_nd':      [(0b0010000000000000, True),    
                             (0b0001000000000000, True)],   
        'snow_ice_low':     [(0b0010000000000000, True),    
                             (0b0001000000000000, False)],  
        'snow_ice_med':     [(0b0010000000000000, False),   
                             (0b0001000000000000, True)],   
        'snow_ice_high':    [(0b0010000000000000, False),   
                             (0b0001000000000000, False)],   
    }

    c1l8lvl1 = {
        'designated_fill':  [(0b0000000000000001, False)],  
        'terrain_occlusion':[(0b0000000000000010, False)],  
        'rad_sat_0':        [(0b0000000000001000, True),    
                             (0b0000000000000100, True)],   
        'rad_sat_12':       [(0b0000000000001000, True),    
                             (0b0000000000000100, False)],  
        'rad_sat_34':       [(0b0000000000001000, False),   
                             (0b0000000000000100, True)],   
        'rad_sat_5':        [(0b0000000000001000, False),   
                             (0b0000000000000100, False)],  
        'cloud':            [(0b0000000000010000, False)],  
        'cloud_nd':         [(0b0000000001000000, True),    
                             (0b0000000000100000, True)],   
        'cloud_low':        [(0b0000000001000000, True),    
                             (0b0000000000100000, False)],  
        'cloud_med':        [(0b0000000001000000, False),   
                             (0b0000000000100000, True)],   
        'cloud_high':       [(0b0000000001000000, False),   
                             (0b0000000000100000, False)],  
        'cloud_shadow_nd':  [(0b0000000100000000, True),    
                             (0b0000000010000000, True)],   
        'cloud_shadow_low': [(0b0000000100000000, True),    
                             (0b0000000010000000, False)],  
        'cloud_shadow_med': [(0b0000000100000000, False),   
                             (0b0000000010000000, True)],   
        'cloud_shadow_high':[(0b0000000100000000, False),   
                             (0b0000000010000000, False)],  
        'snow_ice_nd':      [(0b0000010000000000, True),    
                             (0b0000001000000000, True)],   
        'snow_ice_low':     [(0b0000010000000000, True),    
                             (0b0000001000000000, False)],  
        'snow_ice_med':     [(0b0000010000000000, False),   
                             (0b0000001000000000, True)],   
        'snow_ice_high':    [(0b0000010000000000, False),   
                             (0b0000001000000000, False)],  
        'cirrus_nd':        [(0b0001000000000000, True),    
                             (0b0000100000000000, True)],   
        'cirrus_low':       [(0b0001000000000000, True),    
                             (0b0000100000000000, False)],  
        'cirrus_med':       [(0b0001000000000000, False),   
                             (0b0000100000000000, True)],   
        'cirrus_high':      [(0b0001000000000000, False),   
                             (0b0000100000000000, False)],
    }

    c1l754lvl1 = {
        'designated_fill':  [(0b0000000000000001, False)],  
        'designated_pixel': [(0b0000000000000010, False)],  
        'rad_sat_0':        [(0b0000000000001000, True),    
                             (0b0000000000000100, True)],   
        'rad_sat_12':       [(0b0000000000001000, True),    
                             (0b0000000000000100, False)],  
        'rad_sat_34':       [(0b0000000000001000, False),   
                             (0b0000000000000100, True)],   
        'rad_sat_5':        [(0b0000000000001000, False),   
                             (0b0000000000000100, False)],  
        'cloud':            [(0b0000000000010000, False)],  
        'cloud_nd':         [(0b0000000001000000, True),    
                             (0b0000000000100000, True)],   
        'cloud_low':        [(0b0000000001000000, True),    
                             (0b0000000000100000, False)],  
        'cloud_med':        [(0b0000000001000000, False),   
                             (0b0000000000100000, True)],   
        'cloud_high':       [(0b0000000001000000, False),   
                             (0b0000000000100000, False)],  
        'cloud_shadow_nd':  [(0b0000000100000000, True),    
                             (0b0000000010000000, True)],   
        'cloud_shadow_low': [(0b0000000100000000, True),    
                             (0b0000000010000000, False)],  
        'cloud_shadow_med': [(0b0000000100000000, False),   
                             (0b0000000010000000, True)],   
        'cloud_shadow_high':[(0b0000000100000000, False),   
                             (0b0000000010000000, False)],  
        'snow_ice_nd':      [(0b0000010000000000, True),    
                             (0b0000001000000000, True)],   
        'snow_ice_low':     [(0b0000010000000000, True),    
                             (0b0000001000000000, False)],  
        'snow_ice_med':     [(0b0000010000000000, False),   
                             (0b0000001000000000, True)],   
        'snow_ice_high':    [(0b0000010000000000, False),   
                             (0b0000001000000000, False)],  
    }

    c2l8lvl2 = {
        'fill':             [(0b0000000000000001, False)],  
        'dilated_cloud':    [(0b0000000000000010, False)],  
        'cirrus':           [(0b0000000000000100, False)],   
        'cloud':            [(0b0000000000001000, False)],  
        'cloud_shadow':     [(0b0000000000010000, False)],   
        'snow':             [(0b0000000000100000, False)],  
        'clear':            [(0b0000000001000000, False)],  
        'water':            [(0b0000000010000000, False)],
  
        'cloud_nd':         [(0b0000000100000000, False),    
                             (0b0000001000000000, False)],  
        'cloud_low':        [(0b0000000100000000, True),   
                             (0b0000001000000000, False)],   
        'cloud_med':        [(0b0000000100000000, False),   
                             (0b0000001000000000, True)],  
        'cloud_high':       [(0b0000000100000000, True),    
                             (0b0000001000000000, True)],

        'cloud_shadow_nd':  [(0b0000010000000000, False),    
                             (0b0000100000000000, False)],  
        'cloud_shadow_low': [(0b0000010000000000, True),   
                             (0b0000100000000000, False)],   
        'cloud_shadow_med': [(0b0000010000000000, False),   
                             (0b0000100000000000, True)],  
        'cloud_shadow_high':[(0b0000010000000000, True),    
                             (0b0000100000000000, True)],   

        'snow_ice_nd':      [(0b0001000000000000, False),    
                             (0b0010000000000000, False)],  
        'snow_ice_low':     [(0b0001000000000000, True),   
                             (0b0010000000000000, False)],   
        'snow_ice_med':     [(0b0001000000000000, False),   
                             (0b0010000000000000, True)],
        'snow_ice_high':    [(0b0001000000000000, True),    
                             (0b0010000000000000, True)],

        'cirrus_nd':        [(0b0100000000000000, False),   
                             (0b1000000000000000, False)],   
        'cirrus_low':       [(0b0100000000000000, True),   
                             (0b1000000000000000, False)], 
        'cirrus_med':       [(0b0100000000000000, False),   
                             (0b1000000000000000, True)],   
        'cirrus_high':      [(0b0100000000000000, True),   
                             (0b1000000000000000, True)],  
    }

    c2l7lvl2 = {
        'fill':             [(0b0000000000000001, False)],  
        'dilated_cloud':    [(0b0000000000000010, False)],    
        'cloud':            [(0b0000000000001000, False)],  
        'cloud_shadow':     [(0b0000000000010000, False)],   
        'snow':             [(0b0000000000100000, False)],  
        'clear':            [(0b0000000001000000, False)],  
        'water':            [(0b0000000010000000, False)],
  
        'cloud_nd':         [(0b0000000100000000, False),    
                             (0b0000001000000000, False)],  
        'cloud_low':        [(0b0000000100000000, True),   
                             (0b0000001000000000, False)],   
        'cloud_med':        [(0b0000000100000000, False),   
                             (0b0000001000000000, True)],  
        'cloud_high':       [(0b0000000100000000, True),    
                             (0b0000001000000000, True)],

        'cloud_shadow_nd':  [(0b0000010000000000, False),    
                             (0b0000100000000000, False)],  
        'cloud_shadow_low': [(0b0000010000000000, True),   
                             (0b0000100000000000, False)],   
        'cloud_shadow_med': [(0b0000010000000000, False),   
                             (0b0000100000000000, True)],  
        'cloud_shadow_high':[(0b0000010000000000, True),    
                             (0b0000100000000000, True)],   

        'snow_ice_nd':      [(0b0001000000000000, False),    
                             (0b0010000000000000, False)],  
        'snow_ice_low':     [(0b0001000000000000, True),   
                             (0b0010000000000000, False)],   
        'snow_ice_med':     [(0b0001000000000000, False),   
                             (0b0010000000000000, True)],
        'snow_ice_high':    [(0b0001000000000000, True),    
                             (0b0010000000000000, True)],

    }

    all_flags =dict()
    all_flags[1] = dict()
    all_flags[2] = dict()
    all_flags[1][8] = dict()
    all_flags[2][8] = dict()
    all_flags[2][7] = dict()
    all_flags[2][4] = dict()
    all_flags[2][5] = dict()
    all_flags[2][9] = dict()
    all_flags[1][7] = dict()
    all_flags[1][5] = dict()
    all_flags[1][4] = dict()
    all_flags[1][8][1] = c1l8lvl1
    all_flags[2][8][1] = c2l8lvl1
    all_flags[2][7][1] = c2l7lvl1
    all_flags[1][7][1] = c1l754lvl1
    all_flags[1][5][1] = c1l754lvl1
    all_flags[1][4][1] = c1l754lvl1
    all_flags[2][8][2] = c2l8lvl2
    all_flags[2][7][2] = c2l7lvl2
    all_flags[2][4][2] = c2l7lvl2
    all_flags[2][5][2] = c2l7lvl2
    all_flags[2][9][2] = c2l8lvl2

    return all_flags[collection][ls_number][level]

def MODIS_qa_translator(product_name):
    """Returns a dictionary to bridge between labels and bits in
    MODIS pixel_qa bands.

    Parameters
    ----------
    product_name : str
        For which product to get the values.

    Returns
    -------
    dict
        Keys are labels and values are
        definitions of the bits.
    """

    flag_bits = dict()

    modis_11 = {
        "good_qa":  [(0b00000010, True), (0b00000001, True)],
        "other_qa": [(0b00000010, True), (0b00000001, False)],
        "cloud_qa": [(0b00000010, False), (0b00000001, True)],
        "bad_qa":   [(0b00000010, False), (0b00000001, False)],

        "good_qa2":  [(0b00001000, True),  (0b00000100, True)],
        "other_qa2": [(0b00001000, True),  (0b00000100, False)],
        "TBD":       [(0b00001000, False), (0b00000100, True)],
        "TBD":       [(0b00001000, False), (0b00000100, False)],

        "emis_001":     [(0b00100000, True),  (0b00010000, True)],
        "emis_002":     [(0b00100000, True),  (0b00010000, False)],
        "emis_004":     [(0b00100000, False), (0b00010000, True)],
        "emis_gt_004":  [(0b00100000, False), (0b00010000, False)],

        "lst_1K":    [(0b10000000, True),  (0b01000000, True)],
        "lst_2K":    [(0b10000000, True),  (0b01000000, False)],
        "lst_3K":    [(0b10000000, False), (0b01000000, True)],
        "lst_gt_3K": [(0b10000000, False), (0b01000000, False)],
    }

    flag_bits['MOD11A1.061'] = modis_11
    flag_bits['MYD11A1.061'] = modis_11

    return flag_bits[product_name]

def get_radsat_qa_bits(collection, ls_number, level):
    """Returns a dictionary to bridge between labels and bits in
    Landsat radsat_qa bands.

    Parameters
    ----------
    collection : int
        For which collection to get the values.
    ls_number : int
        For which spacecraft to get the values.
    level : int
        For which level to get the values.

    Returns
    -------
    dict
        Keys are labels (different per Landsat mission/product) and values are
        definitions of the bits.

    Examples
    --------
    # Count from right to left.
    # <-etc...9876543210
    #

    Returns True if 2nd bit is set:
    [(0b0000000000000100, False)]

    Returns True if 9th bit is set AND 8th bit is unset.
    [(0b0000001000000000, False)
     (0b0000000100000000, True)]

    Returns True if 11th bit is unset AND 10th bit is unset.
    [(0b0000100000000000, True)
     (0b0000010000000000, True)]
    """

    c2l7lvl2 = {
        'saturated_band1'  : [(0b0000000000000001, False)],
        'saturated_band2'  : [(0b0000000000000010, False)],
        'saturated_band3'  : [(0b0000000000000100, False)],
        'saturated_band4'  : [(0b0000000000001000, False)],
        'saturated_band5'  : [(0b0000000000010000, False)],
        'saturated_band6L' : [(0b0000000000100000, False)], 
        'saturated_band7'  : [(0b0000000001000000, False)],
        'saturated_band6H' : [(0b0000000100000000, False)],
        'dropped_pixel'    : [(0b0000001000000000, False)],
    }

    c2l8lvl2 = {
        'saturated_band1'  : [(0b0000000000000001, False)],
        'saturated_band2'  : [(0b0000000000000010, False)],
        'saturated_band3'  : [(0b0000000000000100, False)],
        'saturated_band4'  : [(0b0000000000001000, False)],
        'saturated_band5'  : [(0b0000000000010000, False)],
        'saturated_band6'  : [(0b0000000000100000, False)], 
        'saturated_band7'  : [(0b0000000001000000, False)],
        'saturated_band9'  : [(0b0000000100000000, False)],
        'terrain_occlusion': [(0b0000100000000000, False)],
    }

    all_flags =dict()
    all_flags[2] = dict()
    all_flags[2][8] = dict()
    all_flags[2][9] = dict()
    all_flags[2][7] = dict()
    all_flags[2][4] = dict()
    all_flags[2][5] = dict()
    all_flags[2][7][2] = c2l7lvl2
    all_flags[2][4][2] = c2l7lvl2
    all_flags[2][5][2] = c2l7lvl2
    all_flags[2][8][2] = c2l8lvl2
    all_flags[2][9][2] = c2l8lvl2

    return all_flags[collection][ls_number][level]

def PROBAV_qa_translator():
    """Returns a dictionary to bridge between labels and bits in
    PROBAV pixel_qa bands.

    Returns
    -------
    dict
        Keys are labels and values are definitions of the bits.
    """
    flags = {

        'clear':            [(0b0000000000000001, True),
                            (0b0000000000000010, True),
                            (0b0000000000000100, True)],
        'undefined':        [(0b0000000000000001, True),
                            (0b0000000000000010, False),
                            (0b0000000000000100, True)],
        'cloud':            [(0b0000000000000001, True),
                            (0b0000000000000010, False),
                            (0b0000000000000100, False)],
        'ice/snow':         [(0b0000000000000001, False),
                            (0b0000000000000010, True),
                            (0b0000000000000100, True)],
        'shadow':           [(0b0000000000000001, True),    
                            (0b0000000000000010, True),
                            (0b0000000000000100, False)],  

        'sea':              [(0b0000000000001000, True)],

        'bad SWIR':         [(0b0000000000010000, True)],
        'bad NIR':          [(0b0000000000100000, True)],
        'bad RED':          [(0b0000000001000000, True)], 
        'bad BLUE':         [(0b0000000010000000, True)],  
    }
    return flags

def get_mask(qa_array, flags, flag_bits):
    """Given a bitmask (`qa_array`) a list of `flags` to identify and
    a dictionary to convert between the bits and labels, returns a 
    boolean array.

    Parameters
    ----------
    qa_array : np.ndarray or xr.DataArray
        The landsat bitmask
    flags : [type]
        Which classes to look for, e.g. ["cloud", "cloud_shadow"].
    flag_bits : dict
        Keys are labels (defining the valid items in 'flags'), values are bits.
        See 'ls_bitmasks.get_pixel_qa_bits()'.

    Returns
    -------
    np.ndarray or xr.DataArray
        Boolean array with True for pixels belonging to the classes 
        specified in 'flags'.
    """

    if isinstance(qa_array, xr.DataArray):
        final_mask = xr.zeros_like(qa_array)
    else:
        final_mask = np.zeros_like(qa_array)

    for flag in flags:
        all_checks = list()
        for byte, inverse in flag_bits[flag]:
            if inverse:
                all_checks.append(np.bitwise_and(~qa_array, byte) > 0 )
            else:
                all_checks.append(np.bitwise_and(qa_array, byte) > 0 )
        
        if isinstance(qa_array, xr.DataArray):
            flag_mask = xr.concat(all_checks, dim = "checker").all(dim = "checker")
        else:
            flag_mask = np.all(all_checks, axis = 0)
        
        final_mask = final_mask | flag_mask
    
    return final_mask > 0
