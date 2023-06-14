"""
Functions to automatically decorate all the functions inside the et_look_v2 and
et_look_dev submodules so that et_look can be run when not all inputs variables 
are available. This allows to run only parts of et_look, e.g. when only `et_ref_24_mm`
as output is required, pre_et_look doesn't need to download and process all variables.
"""
import types
import numpy as np
from pywapor.general.logger import log

def decorate_mod(module, decorator):
    """Apply a decorator to all the functions inside a module.

    Parameters
    ----------
    module : module
        Module to decorate.
    decorator : function
        Function to decorate with.
    """
    for name in dir(module):
        if name not in [
                        "ra_soil",
                        "initial_friction_velocity_inst",
                        "monin_obukhov_length", 
                        "stability_parameter", 
                        "stability_factor", 
                        "friction_velocity", 
                        "ra_canopy",
                        "calc_dlat_dlon",
                        "calc_slope",
                        "stability_parameter_obs", 
                        "stability_correction_heat_obs"]:
            obj = getattr(module, name)
            if isinstance(obj, types.FunctionType) and not hasattr(obj, 'decorated'):
                setattr(module, name, decorator(obj))

def decorate_submods(module, decorator):
    """Apply a decorator to all the functions inside all the submodules of a 
    module.

    Parameters
    ----------
    module : module
        Module of which the functions inside its submodules to decorate.
    decorator : function
        Function to decorate with.
    """
    for submod in dir(module):
        submod = getattr(module, submod)
        if isinstance(submod, types.ModuleType):
            decorate_mod(submod, decorator)

def etlook_decorator(func):
    """Checks if the DataArrays contain data or are None.

    Parameters
    ----------
    func : function
        Function for which to check the inputs.
    """
    group = str(func.__module__)
    def wrapper_func(*args, **kwargs):
        check1 = np.all([arg.dtype != object for arg in args])
        check2 = np.all([arg.dtype != object for _, arg in kwargs.items()])
        if check1 and check2:
            log.info(f"--> Calculating `{func.__name__}`.")
            x = func(*args, **kwargs)
            x.attrs["calculated_with"] = [arg.name for arg in args]
            x.attrs["et_look_module"] = group
            return x
        else:
            missing_args = [arg.name for arg in args if arg.dtype == object]
            missing_kwargs = [name for name, arg in kwargs.items() if arg.dtype == object]
            missing = missing_args + missing_kwargs
            log.warning(f"--> Insufficient data found for `{func.__name__}`, missing [`{'`, `'.join(missing)}`]")
    wrapper_func.__module__ = func.__module__
    wrapper_func.__name__ = func.__name__
    setattr(wrapper_func, "decorated", True)
    return wrapper_func