import tracemalloc
import datetime
from pywapor.general.logger import log, adjust_logger
import types
import numpy as np
import xarray as xr

def format_bytes(size):
    """Convert bytes to KB, MB, GB or TB.

    Parameters
    ----------
    size : int
        Total bytes.

    Returns
    -------
    tuple
        Converted size and label.
    """
    power = 2**10
    n = 0
    power_labels = {0: '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
    while size > power:
        size /= power
        n += 1
    return size, power_labels[n]+'B'

def performance_check(func):
    """Add memory usage and elapsed time logger to a function.

    Parameters
    ----------
    func : function
        Function to monitor

    Returns
    -------
    function
        Function with added logging and a new `label` keyword argument.
    """
    def wrapper_func(*args, **kwargs):
        if "label" in kwargs.keys():
            label = kwargs.pop("label")
        else:
            label = f"`{func.__module__}.{func.__name__}`"
        log.info(f"--> {label}").add()
        t1 = datetime.datetime.now()
        tracemalloc.start()
        out = func(*args, **kwargs)
        mem_test = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        t2 = datetime.datetime.now()
        size, size_label = format_bytes(mem_test[1]-mem_test[0])
        log.info(f"> peak-memory-usage: {size:.1f}{size_label}, execution-time: {t2-t1}.")
        if isinstance(out, xr.Dataset):
            log.info("> chunksize|dimsize: [" + ", ".join([f"{k}: {v[0]}|{sum(v)}" for k, v in out.unify_chunks().chunksizes.items()]) + "]")
        log.sub()
        return out
    wrapper_func.__module__ = func.__module__
    wrapper_func.__name__ = func.__name__
    setattr(wrapper_func, "decorated", True)
    return wrapper_func

def decorate_function(obj, decorator):
    """Apply a decorator to a function if it hasn't already been decorated by this function.

    Parameters
    ----------
    obj : function
        Function to be decorated.
    decorator : function
        Decorator function.
    """
    module = obj.__module__
    name = obj.__name__
    if isinstance(obj, types.FunctionType) and not hasattr(obj, 'decorated'):
        setattr(module, name, decorator(obj))

@performance_check
def test(n, k = 100):
    x = np.random.random((n,k,1000))**2
    return x
