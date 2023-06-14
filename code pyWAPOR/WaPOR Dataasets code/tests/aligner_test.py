import os
import numpy as np
from pywapor.general import aligner
from pywapor.general.processing_functions import create_dummy_ds
from pywapor.general.logger import adjust_logger

def test_1(tmp_path):

    folder = tmp_path

    adjust_logger(True, folder, "INFO", testing = True)

    test = 0
    enhancers = []
    example_t_vars = ["lst", "bt"]
    chunks = (1, 500, 500)
    
    dss = {
        ("source1", "productX"): create_dummy_ds(["ndvi"], shape = (10, 500, 500), sdate = "2022-02-02", edate = "2022-02-13", fp = os.path.join(folder, f"ndvi_in_test_{test}.nc"), chunks = chunks),
        ("source2", "productY"): create_dummy_ds(["lst"], shape = (16, 100, 100), sdate = "2022-02-01", edate = "2022-02-09", fp = os.path.join(folder, f"lst_in_test_{test}.nc"), min_max = [280, 320], precision=0, chunks = chunks, mask_data = True),
        ("source2", "productZ"): create_dummy_ds(["bt"], shape = (11, 80, 80), sdate = "2022-02-03", edate = "2022-02-14", fp = os.path.join(folder, f"bt_in_test_{test}.nc"), min_max = [290, 330], precision=0, chunks = chunks, mask_data = True),
    }

    sources = {
            "ndvi": {"spatial_interp": "nearest", "temporal_interp": "linear"},
            "lst":  {"spatial_interp": "nearest", "temporal_interp": "linear"},
            "bt":   {"spatial_interp": "nearest", "temporal_interp": "linear"},
                }

    ds = aligner.main(dss, sources, folder, enhancers, example_t_vars = example_t_vars)

    assert ds.rio.crs.to_epsg() == 4326
    assert np.all([{'x': 500, 'y': 500, 'time': 26}[k] == v for k,v in ds.dims.items()])
