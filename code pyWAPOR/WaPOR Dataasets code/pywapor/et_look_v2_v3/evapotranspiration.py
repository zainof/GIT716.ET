from pywapor.et_look_v2_v3 import constants as c
import numpy as np
import xarray as xr

def interception_mm(P_24, vc, lai, int_max=0.2):
    r"""
    Computes the daily interception. The daily interception of a vegetated area
    is calculated according to :footcite:t:`braden1985energiehaushalts` 
    and :footcite:t:`hoyningen1983interception`.

    .. math ::

        I^*=I_{max} \cdot I_{lai} \cdot \left(1-\frac{1}{1+\frac{c_{veg} \cdot P}
        {I_{max} \cdot I_{lai}}}\right)

    Parameters
    ----------
    P_24 :  float
        daily rainfall, 
        :math:`P`
        [mm day :math:`^{-1}`]

    vc : float
        vegetation cover, 
        :math:`c_{veg}`
        [-]

    lai : float
        leaf area index, 
        :math:`I_{lai}`
        [-]

    int_max :  float
        maximum interception per leaf, 
        :math:`I_{max}`
        [mm day :math:`^{-1}`]

    Returns
    -------
    int_mm : float
        interception, 
        :math:`I^*`
        [mm day :math:`^{-1}`]

    """
    if isinstance(lai, xr.DataArray):
        zero_mask = np.logical_or(np.logical_or(vc == 0, lai ==0), P_24 == 0)
    else:
        zero_mask = np.logical_or.reduce((lai == 0, vc == 0, P_24 == 0))

    res = int_max * lai * (1 - (1 / (1 + ((vc * P_24) / (int_max * lai)))))

    if isinstance(res, xr.DataArray):
        res = xr.where(zero_mask, 0.0, res)
    else:
        res[zero_mask] = 0

    return res


def et_reference(rn_24_grass, ad_24, psy_24, vpd_24, ssvp_24, u_24):
    r"""
    Computes the reference evapotranspiration. 
    
    The reference evapotranspiration :math:`ET_{ref}` is an important concept 
    in irrigation science. The reference evapotranspiration can be inferred 
    from routine meteorological measurements. 
    
    The reference evapotranspiration is the evapotranspiration of grass under 
    well watered conditions. First the aerodynamical resistance for 
    grass :math:`r_{a,grass}` [sm :math:`^{-1}`] is calculated:

    .. math ::

        r_{a,grass}=\frac{208}{u_{obs}}

    Then the reference evapotranspiration :math:`ET_{ref}` [W m :math:`^{-2}`] can be calculated
    as follows, with taking the default value for the grass surface resistance
    :math:`r_{grass}` = 70 sm :math:`^{-1}`

    .. math ::

        ET_{ref}=\frac{\Delta \cdot Q_{grass}^{*}+
        \rho c_{p}\frac{\Delta_{e}}{r_{a,grass}}}
        {\Delta+\gamma \cdot \left(1+\frac{r_{grass}}{r_{a,grass}}\right)}

    The soil heat flux is assumed to be zero or close to zero on a daily basis.

    Parameters
    ----------
    rn_24_grass : float
        net radiation for reference grass surface, 
        :math:`Q^{*}_{grass}`
        [Wm-2]
    u_24 : float
        daily wind speed at observation height, 
        :math:`u_{obs}`
        [m/s]
    ad_24 : float
        daily air density, 
        :math:`\rho_{24}`
        [kg m-3]
    psy_24 : float
        daily psychrometric constant, 
        :math:`\gamma_{24}`
        [mbar K-1]
    vpd_24 : float
        daily vapour pressure deficit, 
        :math:`\Delta_{e,24}`
        [mbar]
    ssvp_24 : float
        daily slope of saturated vapour pressure curve, 
        :math:`\Delta_{24}`
        [mbar K-1]

    Returns
    -------
    et_ref_24 : float
        reference evapotranspiration (well watered grass) energy equivalent, 
        :math:`ET_{ref}`
        [W m-2]
    """
    r_grass = 70
    ra_grass = 208. / u_24
    et_ref_24 = (ssvp_24 * rn_24_grass + ad_24 * c.sh * (vpd_24 / ra_grass)) /\
        (ssvp_24 + psy_24 * (1 + r_grass / ra_grass))
    return et_ref_24


def et_reference_mm(et_ref_24, lh_24):
    r"""
    Computes the reference evapotranspiration.

    .. math ::

        ET_{ref}=ET_{ref} \cdot d_{sec} \cdot \lambda_{24}

    where:

    * :math:`d_{sec}` seconds in the day = 86400 [s]

    Parameters
    ----------
    et_ref_24 : float
        daily reference evapotranspiration energy equivalent, 
        :math:`ET_{ref}`
        [W m-2]
    lh_24 : float
        daily latent heat of evaporation, 
        :math:`\lambda_{24}`
        [J/kg]

    Returns
    -------
    et_ref_24_mm : float
        reference evapotranspiration (well watered grass), 
        :math:`ET_{ref}`
        [mm d-1]
    """
    x = et_ref_24 * c.day_sec / lh_24

    if isinstance(x, xr.DataArray):
        et_ref_24_mm = x.clip(0, np.inf)
    else:
        et_ref_24_mm = np.clip(x, 0, np.inf)

    return et_ref_24_mm


def et_actual_mm(e_24_mm, t_24_mm):
    r"""
    Computes the actual evapotranspiration based on the separate calculations
    of evaporation and transpiration.

    .. math ::

        ET = E + T

    Parameters
    ----------
    e_24_mm : float
        daily evaporation in mm, 
        :math:`E`
        [mm d-1]
    t_24_mm : float
        daily transpiration in mm, 
        :math:`T`
        [mm d-1]

    Returns
    -------
    et_24_mm : float
        daily evapotranspiration in mm, 
        :math:`ET`
        [mm d-1]
    """
    return e_24_mm + t_24_mm
