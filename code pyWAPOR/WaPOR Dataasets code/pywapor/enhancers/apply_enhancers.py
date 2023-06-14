from functools import partial

def apply_enhancer(ds, variable, enhancer):
    """Apple a function to a (variable in a) dataset. 

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to enhance.
    variable : str
        Variable name to enhance.
    enhancer : function
        Function to be applied to `variable` inside `ds`. Should take `ds` as
        first argument and `variable` as second.

    Returns
    -------
    tuple
        The enhanced dataset and the label to log when calculating the dataset.
    """
    ds = enhancer(ds, variable)

    if isinstance(enhancer, partial):
        func_name = enhancer.func.__name__
    else:
        func_name = enhancer.__name__
    
    if isinstance(variable, type(None)):
        label = f"--> Applying '{func_name}'."
    else:
        label = f"--> Applying '{func_name}' to `{variable}`."

    return ds, label
