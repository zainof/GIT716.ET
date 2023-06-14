import numpy as np
import json
import datetime
import dateutil.parser
import decimal
import pandas as pd
import pywapor.se_root as se_root
from functools import partial
import types
import importlib

test_out = r"/Users/hmcoerver/Local/test.json"

CONVERTERS = {
    'np.timedelta64': lambda t: np.timedelta64(pd.Timedelta(t)),
    'function': lambda x: getattr(importlib.import_module(x[0]), x[1]),
}

class MyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.timedelta64,)):
            return {"val": str(obj), "_spec_type": "np.timedelta64"}
        elif isinstance(obj, (types.FunctionType,)):
            return {"val": (obj.__module__, obj.__name__), "_spec_type": "function"}
        else:
            return super().default(obj)

def object_hook(obj):
    _spec_type = obj.get('_spec_type')
    if not _spec_type:
        return obj

    if _spec_type in CONVERTERS:
        return CONVERTERS[_spec_type](obj['val'])
    else:
        raise Exception('Unknown {}'.format(_spec_type))

se_root_dler = partial(se_root.se_root, bin_length = "DEKAD", 
                        sources = "level_1")
    
sample = {
    "lst": {
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
                "is_example": True
            },
        ],
        "temporal_interp": {
                            "method": "whittaker",
                            "make_plots": r"/Users/hmcoerver/Local/wt_test2/graphs",
                            "a": 0.99,
                            "valid_drange": [285, 320],
                            "max_dist": np.timedelta64(15, "D"),
                            "lmbdas": 100.,# TODO why does this not work
        },
        "spatial_interp": "nearest",
        },
    "se_root": {
        "products": [
            {
                "source": se_root.se_root,
                "product_name": "v2",
                "enhancers": "default",
            },
        ],
        }
}

# sample = {
#     "test": "blabla",
#     "bla": np.timedelta64(15, "D"),
# }

with open(test_out, 'w') as fp:
    json.dump(sample, fp, cls=MyJSONEncoder)

with open(test_out) as fp:
    ref_sample = json.load(fp, object_hook=object_hook)

print(sample == ref_sample)
