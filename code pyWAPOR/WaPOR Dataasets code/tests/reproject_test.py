from pywapor.general.processing_functions import create_dummy_ds
from pywapor.general.reproject import reproject_bulk, reproject_chunk
from pywapor.general.logger import adjust_logger
import glob
import numpy as np
import os

def test_bulk_vs_chunk(tmp_path):

    folder = tmp_path

    adjust_logger(True, folder, "INFO")
    
    test = 0

    chunks = (1,500,500)

    example_ds = create_dummy_ds(
                                ["ndvi"], 
                                shape = (10, 500, 500), 
                                sdate = "2022-02-02", 
                                edate = "2022-02-13",
                                latlim = [20, 30],
                                lonlim = [40, 50],
                                fp = os.path.join(folder, f"ndvi_in_test_{test}.nc"), 
                                chunks = chunks,
                                data_generator="uniform",
                            )

    src_ds = create_dummy_ds(
                                ["lst"],
                                shape = (16, 100, 100), 
                                sdate = "2022-02-01", 
                                edate = "2022-02-09",
                                latlim = [15, 25],
                                lonlim = [45, 55],
                                fp = os.path.join(folder, f"lst_in_test_{test}.nc"), 
                                min_max = [280, 320], 
                                precision=0, 
                                chunks = chunks,
                                data_generator="uniform",
                                mask_data = True
                            )

    dst_path = os.path.join(folder, f"out_{test}.nc")
    spatial_interp = "nearest"
    ds_chunk = reproject_chunk(src_ds, example_ds, dst_path, spatial_interp = spatial_interp)
    assert ds_chunk.rio.crs.to_epsg() == 4326
    assert np.all([example_ds.dims[k] == v for k,v in ds_chunk.dims.items() if k != "time"])
    assert ds_chunk.rio.bounds() == example_ds.rio.bounds()

    test = 1
    dst_path = os.path.join(folder, f"out_{test}.nc")
    spatial_interp = "nearest"
    ds_bulk = reproject_bulk(src_ds, example_ds, dst_path, spatial_interp = spatial_interp)
    assert ds_bulk.rio.crs.to_epsg() == 4326
    assert np.all([example_ds.dims[k] == v for k,v in ds_bulk.dims.items() if k != "time"])
    assert ds_bulk.rio.bounds() == example_ds.rio.bounds()

    assert ds_bulk.equals(ds_chunk)
    assert np.isclose(ds_bulk.lst.mean().values, ds_chunk.lst.mean().values)
    assert np.all(ds_bulk.lst.isnull().sum("time").values == ds_chunk.lst.isnull().sum("time").values)
