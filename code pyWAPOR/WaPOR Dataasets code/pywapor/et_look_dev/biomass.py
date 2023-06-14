# -*- coding: utf-8 -*-

from pywapor.et_look_dev import constants as c

def lue(lue_max, stress_temp, stress_moist, eps_a):
    r"""
    Computes the light use efficiency

    Parameters
    ----------
    lue_max : float
        maximal light use efficiency
        :math:`EF_{ref}`
        [gr/MJ]
    stress_temp : float
        stress factor for air temperature
        :math:`S_{T}`
        [-]
    stress_moist : float
        stress soil moisture
        :math:`I`
        [-]
    eps_a : float
        epsilon autotrophic respiration
        :math:`I`
        [-]
        
    Returns
    -------
    lue : float
        light use efficiency
        :math:`I`
        [gr/MJ]
    """   
    
    lue = lue_max * stress_temp * stress_moist * eps_a

    return lue
   
def biomass(apar, lue):     
    r"""
    Computes the light use efficiency

    Parameters
    ----------
    apar : float
        apar
        :math:`I_{ra_24,fpar}`
        [MJ/m2]
    lue : float
        light use efficiency
        :math:`I`
        [gr/MJ]
        
    Returns
    -------
    biomass : float
        biomass production
        :math:`I`
        [kg/ha]
    """  
    
    biomass = apar * lue * 10
    
    return biomass