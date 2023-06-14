import os
import numpy as np
from pywapor.general import compositer
from pywapor.general.processing_functions import create_dummy_ds
from pywapor.general.logger import adjust_logger
import datetime

def test_1(tmp_path):

    folder = tmp_path

    adjust_logger(True, folder, "INFO", testing = True)

    test = 0
    enhancers = []
    chunks = (1, 500, 500)
    
    dss = {
        ("source1", "productX"):    create_dummy_ds(["ndvi"], shape = (10, 400, 400), sdate = "2022-02-02", edate = "2022-03-13", fp = os.path.join(folder, f"INPUT/ndvi_in_test_{test}.nc"), chunks = chunks, data_generator="uniform"),
        ("source2", "productY"):    create_dummy_ds(["r0"], shape = (16, 100, 100), sdate = "2022-02-12", edate = "2022-03-09", fp = os.path.join(folder, f"INPUT/r01_in_test_{test}.nc"), min_max = [280, 320], precision=0, chunks = chunks, mask_data = True, data_generator="uniform"),
        ("source2", "productYY"):   create_dummy_ds(["r0"], shape = (16, 100, 100), sdate = "2022-02-12", edate = "2022-03-09", fp = os.path.join(folder, f"INPUT/r02_in_test_{test}.nc"), min_max = [280, 320], precision=0, chunks = chunks, mask_data = True, data_generator="uniform").isel(time=slice(0, -1, 2)),
        ("source3", "productZ"):    create_dummy_ds(["z"], shape = (1, 80, 80), sdate = "2022-02-01", edate = "2022-03-14", fp = os.path.join(folder, f"INPUT/z1_in_test_{test}.nc"), min_max = [290, 330], precision=0, chunks = chunks, mask_data = True, data_generator="uniform").isel(time=0, drop = True),
        ("source4", "productA"):    create_dummy_ds(["z"], shape = (2, 80, 80), sdate = "2022-02-01", edate = "2022-03-14", fp = os.path.join(folder, f"INPUT/z2_in_test_{test}.nc"), min_max = [290, 330], precision=0, chunks = chunks, mask_data = True, data_generator="uniform"),
        ("source5", "productB"):    create_dummy_ds(["t_opt"], shape = (1, 90, 80), sdate = "2022-02-01", edate = "2022-03-14", fp = os.path.join(folder, f"INPUT/topt1_in_test_{test}.nc"), min_max = [290, 330], precision=0, chunks = chunks, mask_data = True, data_generator="uniform").isel(time=0, drop = True),
        ("source6", "productC"):    create_dummy_ds(["t_opt"], shape = (1, 80, 90), sdate = "2022-02-01", edate = "2022-03-14", fp = os.path.join(folder, f"INPUT/topt2_test_{test}.nc"), min_max = [290, 330], precision=0, chunks = chunks, mask_data = True, data_generator="uniform").isel(time=0, drop = True),
    }

    sources = {
            "ndvi": {"spatial_interp": "nearest", "temporal_interp": "linear", "composite_type": "mean"},
            "r0":  {"spatial_interp": "nearest", "temporal_interp": "linear", "composite_type": "mean"},
            "z":   {"spatial_interp": "nearest", "temporal_interp": "linear", "composite_type": "mean"},
            "t_opt":   {"spatial_interp": "average", "temporal_interp": None, "composite_type": "mean"},
                }

    bins = compositer.time_bins([datetime.datetime(2022, 2, 1), datetime.datetime(2022, 3, 11)], "DEKAD")

    ds = compositer.main(dss, sources, folder, enhancers, bins)
    assert ds.rio.crs.to_epsg() == 4326
    assert np.all([{'x': 400, 'y': 400, 'time_bins': 4}[k] == v for k,v in ds.dims.items()])
    assert "time_bins" not in ds.t_opt.dims