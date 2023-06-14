# import numpy as np

# def get_units(dss):
#     """Get an overview of the units of different variables inside xr.Datasets.

#     Parameters
#     ----------
#     dss : list
#         List with xr.Datasets. Each variable in each dataset is checked for a 
#         `"unit"` attribute.

#     Returns
#     -------
#     dict
#         Dictionary with the units per variable.
#     """
#     units = dict()
#     for sub_ds in dss:
#         variables = list(sub_ds.keys())
#         for var in variables:
#             if hasattr(sub_ds[var], "unit"):
#                 unit = sub_ds[var].unit
#             else:
#                 unit = "unknown"
#             if var in units.keys():
#                 units[var] = np.append(units[var], unit)
#             else:
#                 units[var] = np.array([unit])
#     return units

# def check_units(units, strictness = "low"):
#     """Test whether each variable (as key in `units`) has identical units.

#     Parameters
#     ----------
#     units : dict
#         Dictionary with the units per variable, can be generated with `compositer.get_units`.
#     strictness : {"low" | "medium" | "high}, optional
#         low - Units need to be the same, but 'unknown' units are assumed to be correct.
#         med - Units need to be the same and 'unknown' units are assumed to be different.
#         high - All units must be known and identical.

#     Examples
#     --------
#     >>> units = {"test1": np.array(["C", "C","unknown"]),
#     ...          "test2": np.array(["C", "C", "K"]),
#     ...          "test3": np.array(["C", "C", "C"]),
#     ...          "test4": np.array(["unknown", "unknown", "unknown"])}
#     >>> check_units(units)
#     {'test1': True, 'test2': False, 'test3': True, 'test4': True}
#     >>> check_units(units, strictness = "med")
#     {'test1': False, 'test2': False, 'test3': True, 'test4': True}
#     >>> check_units(units, strictness = "high")
#     {'test1': False, 'test2': False, 'test3': True, 'test4': False}
#     """
#     results = dict()
#     if strictness == "low":
#         for k, v in units.items():
#             check = np.unique(v[v!="unknown"]).size <= 1
#             results[k] = check
#     if strictness == "med":
#         for k, v in units.items():
#             check = np.unique(v).size == 1
#             results[k] = check
#     if strictness == "high":
#         for k, v in units.items():
#             check = np.unique(v).size == 1 and "unknown" not in v
#             results[k] = check
#     return results