import numpy as np
from pywapor.et_look_v2_v3 import constants as con
import xarray as xr
import warnings

def longitude_rad(lon_deg):
    r"""
    Converts longitude from degrees to radians.

    Parameters
    ----------
    lon_deg : float
        longitude in degrees, 
        :math:`\phi`
        [deg]

    Returns
    -------
    lon : float
        longitude, 
        :math:`\phi`
        [rad]

    """
    return lon_deg * np.pi/180.0


def latitude_rad(lat_deg):
    r"""
    Converts latitude from degrees to radians.

    Parameters
    ----------
    lat_deg : float
        latitude in degrees, 
        :math:`\lambda`
        [deg]

    Returns
    -------
    lat : float
        latitude, 
        :math:`\lambda`
        [rad]

    """
    return lat_deg * np.pi/180.0


def slope_rad(slope_deg):
    r"""
    Converts slope from degrees to radians.

    Parameters
    ----------
    slope_deg : float
        slope in degrees, 
        :math:`s`
        [deg]

    Returns
    -------
    slope : float
        slope, 
        :math:`\Delta`
        [rad]

    """
    return slope_deg * np.pi/180.0


def aspect_rad(aspect_deg):
    r"""
    Converts aspect from degrees to radians.

    Parameters
    ----------
    aspect_deg : float
        aspect in degrees, 
        :math:`s`
        [deg]

    Returns
    -------
    aspect : float
        aspect (0 is north; pi is south), 
        :math:`\alpha`
        [rad]
    """
    return aspect_deg * np.pi/180.0


def declination(doy):
    r"""
    Computes the solar declination which is the angular height of the sun
    above the astronomical equatorial plane in radians.

    .. math ::

        \delta=0.409 \cdot \sin\left(\frac{2\pi \cdot J}{365}-1.39\right)

    Parameters
    ----------
    doy : float
        julian day of the year, 
        :math:`J`
        [-]

    Returns
    -------
    decl : float
        declination, 
        :math:`\delta`
        [rad]

    Examples
    --------
    >>> import ETLook.solar_radiation as solrad
    >>> solrad.declination(180)
    0.40512512455439242
    """
    return 0.409 * np.sin(2 * np.pi / 365 * doy - 1.39)


def earth_sun_distance(doy):
    r"""
    Computes the earth sun distance in Angstrom Unit where 1 AU is 1.496e8 km.

    .. math ::

        d_{r}=1+0.033 \cdot \cos\left(\frac{2\pi \cdot J}{365}\right)

    Parameters
    ----------
    doy : float
        julian day of the year, 
        :math:`J`
        [-]

    Returns
    -------
    ed : float
        earth sun distance, 
        :math:`d_{r}`
        [AU]

    Examples
    --------
    >>> import ETLook.solar_radiation as solrad
    >>> solrad.earth_sun_distance(180)
    0.96703055420162642
    """

    warnings.warn(('Using "ed" for the inverse earth-sun distance is deprecated, '
                   'use "iesd" (solar_radiation.inverse_earth_sun_distance) instead'),
                  DeprecationWarning, stacklevel=2)

    return 1 + 0.033 * np.cos(doy * 2 * np.pi / 365.0)


def inverse_earth_sun_distance(doy):
    r"""
    Computes the inverse earth sun distance (iesd) in Angstrom Unit where 1 AU is 1.496e8 km.

    .. math ::

        d_{r}=1+0.033 \cdot \cos\left(\frac{2\pi \cdot J}{365}\right)

    Parameters
    ----------
    doy : float
        julian day of the year, 
        :math:`J`
        [-]

    Returns
    -------
    iesd : float
        inverse earth sun distance, 
        :math:`d_{r}`
        [AU]

    Examples
    --------
    >>> import ETLook.solar_radiation as solrad
    >>> solrad.inverse_earth_sun_distance(180)
    0.96703055420162642
    """

    return 1 + 0.033 * np.cos(doy * 2 * np.pi / 365.0)


def actual_earth_sun_distance(iesd):
    r"""
    Computes the earth sun distance (esd) in Angstrom Unit where 1 AU is 1.496e8 km.

    .. math ::

        d_{r}=1+0.033 \cdot \cos\left(\frac{2\pi \cdot J}{365}\right)

    Parameters
    ----------
    iesd : float
        inverse earth sun distance, 
        :math:`J`
        [AU]

    Returns
    -------
    esd : float
        earth sun distance, 
        :math:`d_{r}`
        [AU]

    Examples
    --------
    >>> import ETLook.solar_radiation as solrad
    >>> solrad.actual_earth_sun_distance(180)
    1.034093489244084
    """

    return 1 / iesd


def seasonal_correction(doy):
    r"""
    Computes the seasonal correction for solar time  in hours.

    .. math ::

        b=\frac{2\pi\left(J-81\right)}{365}

    .. math ::

        s_{c}= 0.1645 \cdot sin \left( 2b \right) - 0.1255 \cdot cos \left(b \right) - 0.025 \left(b\right)

    Parameters
    ----------
    doy : float
        julian day of the year, 
        :math:`J`
        [-]

    Returns
    -------
    sc : float
        seasonal correction, 
        :math:`s_{c}`
        [hours]

    Examples
    --------
    >>> import ETLook.solar_radiation as solrad
    >>> solrad.seasonal_correction(180)
    -0.052343379605521212
    """
    b = 2 * np.pi * (doy - 81) / 365.0
    return 0.1645 * np.sin(2 * b) - 0.1255 * np.cos(b) - 0.025 * np.sin(b)


def sunset_hour_angle(lat, decl):
    r"""
    Computes the sunset hour angle.

    .. math ::

        w_{s}=\arccos(-\tan(\lambda)\cdot \tan(\delta))

    Parameters
    ----------
    decl : float
        solar declination, 
        :math:`\delta`
        [rad]
    lat : float
        latitude, 
        :math:`\lambda`
        [rad]

    Returns
    -------
    ws : float
        sunset hour angle, 
        :math:`w_{s}`
        [rad]

    """
    return np.arccos(-(np.tan(lat) * np.tan(decl)))


def hour_angle(sc, dtime, lon = 0):
    r"""
    Computes the hour angle which is zero at noon and -pi at 0:00 am and
    pi at 12:00 pm

    .. math ::

        \omega=\left(\frac{\pi}{12}\right)\cdot \left(t+s_{c}-12\right)

    Parameters
    ----------
    sc : float
        seasonal correction, 
        :math:`s_{c}`
        [hours]
    dtime : float
        decimal time, 
        :math:`t`
        [hours]
    lon : float
        longitude, 
        :math:`\phi`
        [rad]

    Returns
    -------
    ha : float
        hour_angle, 
        :math:`\omega`
        [rad]

    Examples
    --------
    >>> import ETLook.solar_radiation as solrad
    >>> solrad.hour_angle(sc=solrad.seasonal_correction(75), dtime=11.4)
    -0.19793970172084141
    """
    dtime = dtime + (lon / (15*np.pi/180.0))
    return (np.pi / 12.0) * (dtime + sc - 12.0)


def inst_solar_radiation_toa(csza, iesd):
    r"""
    Computes the instantaneous solar radiation at the top of
    the atmosphere [Wm-2].

    .. math ::

        S_{toa}^{i} = S_{sun} \cdot d_{r} \cdot \phi

    Parameters
    ----------
    csza : float
        cosine solar zenith angle, 
        :math:`\phi`
        [-]
    iesd : float
        inverse earth sun distance, 
        :math:`d_{r}`
        [AU]

    Returns
    -------
    ra_i_toa : float
        instantaneous solar radiation at top of atmosphere, 
        :math:`S_{toa}^{i}`
        [Wm-2]

    Examples
    --------
    >>> import ETLook.solar_radiation as solrad
    >>> doy = 1
    >>> sc = solrad.seasonal_correction(doy)
    >>> ha = solrad.hour_angle(sc, dtime=12)
    >>> decl = solrad.declination(doy)
    >>> csza = solrad.cosine_solar_zenith_angle(ha, decl, 0)
    >>> iesd = solrad.inverse_earth_sun_distance(doy)
    >>> solrad.inst_solar_radiation_toa(csza, iesd)
    1299.9181944414036
    """
    return csza * con.sol * iesd


def daily_solar_radiation_toa(sc, decl, iesd, lat, slope=0, aspect=0):
    r"""
    Computes the daily solar radiation at the top of the atmosphere.

    .. math ::

        S_{toa}=S_{sun} \cdot d_{r}\int_{i=-\pi}^{i=\pi}S_{toa}^{i}

    Parameters
    ----------
    iesd : float
        inverse earth sun distance, 
        :math:`d_{r}`
        [AU]
    decl : float
        solar declination, 
        :math:`\delta`
        [rad]
    sc : float
        seasonal correction, 
        :math:`s_{c}`
        [hours]
    lat : float
        latitude, 
        :math:`\lambda`
        [rad]
    slope : float
        slope, 
        :math:`\Delta`
        [rad]
    aspect : float
        aspect (0 is north; pi is south), 
        :math:`\alpha`
        [rad]

    Returns
    -------
    ra_24_toa : float
        daily solar radiation at the top of atmosphere, 
        :math:`S_{toa}`
        [Wm-2]

    Examples
    --------
    >>> import ETLook.solar_radiation as solrad
    >>> from math import pi
    >>> doy = 1
    >>> sc = solrad.seasonal_correction(doy)
    >>> decl = solrad.declination(doy)
    >>> iesd = solrad.inverse_earth_sun_distance(doy)
    >>> solrad.daily_solar_radiation_toa(sc, decl, iesd, lat=25*pi/180.0)
    265.74072308978026
    """

    # hour angle for the whole day in half-hourly intervals (0:15-23:45)
    t_start = 0.25
    t_end = 24.00
    interval = 0.5
    times = [t_start+i*interval for i in range(0, 48)]
    hours = [hour_angle(sc, t) for t in times]

    ra24 = 0

    for t in hours:
        csza = cosine_solar_zenith_angle(t, decl, lat, slope, aspect)
        ra24 += inst_solar_radiation_toa(csza, iesd) / len(hours)

    # return the average daily radiation
    return ra24


def cosine_solar_zenith_angle(ha, decl, lat, slope=0, aspect=0):
    r"""
    computes the cosine of the solar zenith angle [-].

    .. math ::

        \phi = & \sin\left(\delta\right) \cdot \sin\left(\lambda\right) \cdot  \cos\left(\Delta\right) - \\
        & \sin\left(\delta\right) \cdot \cos\left(\lambda\right) \cdot \sin\left(\Delta\right) + \\
        & \cos\left(\delta\right) \cdot \cos\left(\lambda\right) \cdot \cos\left(\Delta\right) \cdot \cos\left(\omega\right)+\\
        & \cos\left(\delta\right) \cdot \sin\left(\lambda\right) \cdot \sin\left(\Delta\right) \cdot \sin\left(\alpha\right) \cdot \cos\left(\omega\right)+\\
        & \cos\left(\delta\right) \cdot \sin\left(\Delta\right) \cdot  \sin\left(\alpha\right) \cdot \sin\left(\omega\right)

    Parameters
    ----------
    ha : float
        hour angle, 
        :math:`\omega`
        [rad]
    decl : float
        declination, 
        :math:`\delta`
        [rad]
    lat : float
        latitude, 
        :math:`\lambda`
        [rad]
    slope : float
        slope, 
        :math:`\Delta`
        [rad]
    aspect : float
        aspect (0 is north; pi is south), 
        :math:`\alpha`
        [rad]

    Returns
    -------
    csza : float
        cosine solar zenith angle, 
        :math:`\phi`
        [-]

    Examples
    --------
    >>> import ETLook.solar_radiation as solrad
    >>> sc = solrad.seasonal_correction(1)
    >>> ha = solrad.hour_angle(sc, dtime=12)
    >>> solrad.cosine_solar_zenith_angle(ha, decl=solrad.declination(1), lat=0)
    0.92055394167363314
    """
    t1 = np.sin(decl) * np.sin(lat) * np.cos(slope)
    t2 = np.sin(decl) * np.cos(lat) * np.sin(slope) * np.cos(aspect - np.pi)
    t3 = np.cos(decl) * np.cos(lat) * np.cos(slope)
    t4 = np.cos(decl) * np.sin(lat) * np.sin(slope) * np.cos(aspect - np.pi)
    t5 = np.cos(decl) * np.sin(slope) * np.sin(aspect - np.pi)
    csza = t1 - t2 + t3 * np.cos(ha) + t4 * np.cos(ha) + t5 * np.sin(ha)

    # check if the sun is above the horizon
    check = np.sin(decl) * np.sin(lat) + np.cos(decl) * \
        np.cos(lat) * np.cos(ha)

    if isinstance(csza, xr.DataArray):
        res = xr.where((csza > 0) & (check >= 0), csza, 0)
        res = xr.where(csza.notnull() | check.notnull(), res, np.nan)
    else:
        nans = np.logical_or(np.isnan(csza), np.isnan(check))
        res = np.where(np.logical_and(csza > 0, check >= 0), csza, 0)
        res[nans] = np.nan

    return res


def transmissivity(ra, ra_flat):
    """Computes the transmissivity.

    Parameters
    ----------
    ra_24 : float
        daily solar radiation for a flat surface, 
        :math:`S^{\downarrow}`
        [Wm-2]
    ra_24_toa_flat : float
        daily solar radiation at the top of atmosphere for a flat surface, 
        :math:`S_{toa,f}`
        [Wm-2]

    Returns
    -------
    trans_24 : float
        daily atmospheric transmissivity, 
        :math:`\tau`
        [-]
    """
    return ra / ra_flat


def daily_solar_radiation_toa_flat(decl, iesd, lat, ws):
    r"""
    Computes the daily solar radiation at the top of the atmosphere for a flat
    surface.

    .. math ::

        S_{toa,f}=\frac{S_{sun}}{\pi} \cdot d_{inv,r} \cdot (w_{s} \cdot \sin(\lambda) \cdot \sin(\delta) +
                  \cos(\lambda)\cdot\cos(\delta)\cdot\sin(w_{s}))

    Parameters
    ----------
    decl : float
        solar declination, 
        :math:`\delta`
        [rad]
    iesd : float
        inverse earth sun distance, 
        :math:`d_{inv,r}`
        [AU]
    lat : float
        latitude, 
        :math:`\lambda`
        [rad]
    ws : float
        sunset hour angle, 
        :math:`w_{s}`
        [rad]

    Returns
    -------
    ra_24_toa_flat : float
        daily solar radiation at the top of atmosphere for a flat surface, 
        :math:`S_{toa,f}`
        [Wm-2]

    """
    ra_flat = (con.sol / np.pi) * iesd * (ws * np.sin(lat) * np.sin(decl) +
                                        np.cos(lat) * np.cos(decl) * np.sin(ws))

    return ra_flat


def daily_solar_radiation_flat(ra_24_toa_flat, trans_24):
    r"""
    Computes the daily solar radiation at the earth's surface.

    .. math ::

        S^{\downarrow} = \tau \cdot S_{toa}

    Parameters
    ----------
    ra_24_toa_flat : float
        daily solar radiation at the top of atmosphere for a flat surface, 
        :math:`S_{toa}`
        [Wm-2]
    trans_24 : float
        daily atmospheric transmissivity, 
        :math:`\tau`
        [-]

    Returns
    -------
    ra_24 : float
        daily solar radiation for a flat surface, 
        :math:`S^{\downarrow}`
        [Wm-2]

    """
    return ra_24_toa_flat * trans_24


def diffusion_index(trans_24, diffusion_slope=-1.33, diffusion_intercept=1.15):
    r"""
    Computes the diffusion index, the ratio between diffuse and direct
    solar radiation. The results are clipped between 0 and 1.

    .. math ::

        I_{diff} = a_{diff}+b_{diff} \cdot \tau

    Parameters
    ----------
    trans_24 : float
        daily atmospheric transmissivity, 
        :math:`\tau`
        [-]
    diffusion_slope : float
        slope of diffusion index vs transmissivity relationship, 
        :math:`b_{diff}`
        [-]
    diffusion_intercept : float
        intercept of diffusion index vs transmissivity relationship, 
        :math:`a_{diff}`
        [-]

    Returns
    -------
    diffusion_index : float
        diffusion_index, 
        :math:`I_{diff}`
        [-]

    """
    res = diffusion_intercept + trans_24 * diffusion_slope

    res = np.clip(res, 0, 1)

    return res


def daily_total_solar_radiation(ra_24_toa, ra_24_toa_flat, diffusion_index, trans_24):
    r"""
    Computes the daily solar radiation at the earth's surface taken
    diffuse and direct solar radiation into account.

    .. math ::

        S^{\downarrow} = I_{diff} \cdot \tau \cdot S_{toa,f} +(1-I_{diff}) \cdot \tau \cdot S_{toa}

    Parameters
    ----------
    ra_24_toa : float
        daily solar radiation at the top of atmosphere, 
        :math:`S_{toa}`
        [Wm-2]
    ra_24_toa_flat : float
        daily solar radiation at the top of atmosphere for a flat surface, 
        :math:`S_{toa,f}`
        [Wm-2]
    diffusion_index : float
        diffusion_index, 
        :math:`I_{diff}`
        [-]
    trans_24 : float
        daily atmospheric transmissivity, 
        :math:`\tau`
        [-]

    Returns
    -------
    ra_24 : float
        daily solar radiation, 
        :math:`S^{\downarrow}`
        [Wm-2]

    """
    diffuse = trans_24 * ra_24_toa_flat * diffusion_index
    direct = trans_24 * ra_24_toa * (1 - diffusion_index)
    return diffuse + direct

