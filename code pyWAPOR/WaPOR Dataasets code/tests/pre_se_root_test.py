import pywapor
import datetime
from pywapor.general.logger import adjust_logger
import numpy as np
import os

def test_1(tmp_path):

    folder = tmp_path
    adjust_logger(True, folder, "INFO", testing = True)

    sources = dict()
    sources["ndvi"] = {'products': [{'source': 'SENTINEL2',
        'product_name': 'S2MSI2A_R20m',
        'enhancers': 'default',
        'is_example': True}],
    'temporal_interp': 'linear',
    'spatial_interp': 'nearest'}

    sources["lst"] = {'products': [{'source': 'SENTINEL3',
        'product_name': 'SL_2_LST___',
        'enhancers': 'default'}],
    'temporal_interp': 'linear',
    'spatial_interp': 'nearest'}

    sources["bt"] = {'products': [{'source': 'VIIRSL1',
        'product_name': 'VNP02IMG',
        'enhancers': 'default'}],
    'temporal_interp': 'linear',
    'spatial_interp': 'nearest'}

    latlim = [29.4, 29.7]
    lonlim = [30.7, 31.0]
    timelim = [datetime.date(2023, 2, 1), datetime.date(2023, 2, 11)]
    bin_length = "DEKAD"
    ds = pywapor.pre_se_root.main(folder, latlim, lonlim, timelim, sources, bin_length = bin_length)
    assert ds.rio.crs.to_epsg() == 4326
    assert "bt" not in ds.data_vars
    assert "lst" in ds.data_vars
    assert "ndvi" in ds.data_vars
    assert ds.ndvi.min().values >= -1.
    assert ds.ndvi.max().values <= 1.
    assert 240 < ds.lst.mean().values < 330

def test_2(tmp_path):

    folder = tmp_path
    adjust_logger(True, folder, "INFO", testing = True)

    timelim = [datetime.date(2023, 2, 1), datetime.date(2023, 2, 11)]
    latlim = [29.4, 29.6]
    lonlim = [30.7, 30.9]
    sources = "level_2_v3"
    bin_length = 3
    enhancers = []
    buffer_timelim = True
    input_data = pywapor.pre_se_root.main(folder, latlim, lonlim, timelim, 
                                            sources, bin_length = bin_length, 
                                            enhancers = enhancers, buffer_timelim = buffer_timelim)
    assert input_data.rio.crs.to_epsg() == 4326
    assert np.all([x in input_data.data_vars for x in ["ndvi", "p_air_i", "p_air_0_i", "r0_bare", "r0_full", "t_air_i", "t_dew_i", "u_i", "wv_i", "lst"]])
    assert input_data.ndvi.min().values >= -1.
    assert input_data.ndvi.max().values <= 1.
    assert 90 < input_data.p_air_i.mean().values < 110
    assert 90 < input_data.p_air_0_i.mean().values < 110
    assert 0 < input_data.r0_bare.mean().values < 1
    assert 0 < input_data.r0_full.mean().values < 1
    assert -40 < input_data.t_air_i.mean().values < 50
    assert -40 < input_data.t_dew_i.mean().values < 50
    assert 0 < input_data.u_i.mean().values < 150
    assert 0 < input_data.wv_i.mean().values < 100
    assert 240 < input_data.lst.mean().values < 320

    ds = pywapor.se_root.main(input_data, se_root_version = "v3")
    assert ds.rio.crs.to_epsg() == 4326
    assert "se_root" in ds.data_vars
    assert ds.se_root.min().values >= 0.
    assert ds.se_root.max().values <= 1.

def test_3(tmp_path):

    folder = tmp_path
    adjust_logger(True, folder, "INFO", testing = True)

    timelim = [datetime.date(2019, 4, 1), datetime.date(2019, 4, 11)]
    latlim = [29.4, 29.6]
    lonlim = [30.7, 30.9]
    sources = "level_1"
    enhancers = []
    bin_length = "DEKAD"
    input_data = pywapor.pre_se_root.main(folder, latlim, lonlim, timelim, sources, 
                                          bin_length = bin_length, enhancers = enhancers)
    assert input_data.rio.crs.to_epsg() == 4326
    assert np.all([x in input_data.data_vars for x in ["ndvi", "lst"]])
    assert input_data.ndvi.min().values >= -1.
    assert input_data.ndvi.max().values <= 1.
    assert 90 < input_data.p_air_i.mean().values < 110
    assert 90 < input_data.p_air_0_i.mean().values < 110
    assert 0 < input_data.r0_bare.mean().values < 1
    assert 0 < input_data.r0_full.mean().values < 1
    assert -40 < input_data.t_air_i.mean().values < 50
    assert 0 < input_data.wv_i.mean().values < 100
    assert 240 < input_data.lst.mean().values < 320
    
    ds = pywapor.se_root.main(input_data, se_root_version = "v2")
    assert ds.rio.crs.to_epsg() == 4326
    assert "se_root" in ds.data_vars
    assert ds.se_root.min().values >= 0.
    assert ds.se_root.max().values <= 1.

def test_4(tmp_path):

    folder = tmp_path
    adjust_logger(True, folder, "INFO", testing = True)

    timelim = [datetime.date(2019, 4, 1), datetime.date(2019, 4, 3)]
    latlim = [29.4, 29.6]
    lonlim = [30.7, 30.9]
    sources = "level_2"
    input_data = pywapor.pre_se_root.main(folder, latlim, lonlim, timelim, 
                                            sources, bin_length = 3)
    assert input_data.rio.crs.to_epsg() == 4326
    assert np.all([x in input_data.data_vars for x in ["ndvi", "lst"]])
    assert input_data.ndvi.min().values >= -1.
    assert input_data.ndvi.max().values <= 1.
    assert 90 < input_data.p_air_i.mean().values < 110
    assert 90 < input_data.p_air_0_i.mean().values < 110
    assert 0 < input_data.r0_bare.mean().values < 1
    assert 0 < input_data.r0_full.mean().values < 1
    assert -40 < input_data.t_air_i.mean().values < 50
    assert 0 < input_data.wv_i.mean().values < 100
    assert 240 < input_data.lst.mean().values < 320

    ds = pywapor.se_root.main(input_data, se_root_version = "v2")
    assert ds.rio.crs.to_epsg() == 4326
    assert "se_root" in ds.data_vars
    assert ds.se_root.min().values >= 0. 
    assert ds.se_root.max().values <= 1.

def test_5(tmp_path):

    folder = tmp_path
    adjust_logger(True, folder, "INFO", testing = True)

    timelim = [datetime.date(2019, 10, 1), datetime.date(2019, 10, 11)]
    latlim = [29.4, 29.6]
    lonlim = [30.7, 30.9]
    sources = "level_3"
    input_data = pywapor.pre_se_root.main(folder, latlim, lonlim, timelim, 
                                            sources, bin_length = 3)
    assert input_data.rio.crs.to_epsg() == 4326
    assert np.all([x in input_data.data_vars for x in ["ndvi", "lst"]])
    assert input_data.ndvi.min().values >= -1.
    assert input_data.ndvi.max().values <= 1.
    assert 90 < input_data.p_air_i.mean().values < 110
    assert 90 < input_data.p_air_0_i.mean().values < 110
    assert 0 < input_data.r0_bare.mean().values < 1
    assert 0 < input_data.r0_full.mean().values < 1
    assert -40 < input_data.t_air_i.mean().values < 50
    assert 0 < input_data.wv_i.mean().values < 100
    assert 240 < input_data.lst.mean().values < 320

    ds = pywapor.se_root.main(input_data, se_root_version = "v2")
    assert ds.rio.crs.to_epsg() == 4326
    assert "se_root" in ds.data_vars
    assert ds.se_root.min().values >= 0.
    assert ds.se_root.max().values <= 1.