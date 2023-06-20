"""
 Code to run the default pywapor model
 Needs NASA Earthdata login details collecting MODIS and MERRA2 datasets

"""
from osgeo import gdal
import pywapor
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np


def create_et_look_input():
    pywapor.collect.accounts.setup("NASA")

    # User inputs
    # Set the project folder
    project_folder = r""
    # Time period: default composite length of pyWAPOR 10 days
    timelim = ["2023-01-01", "2023-01-31"]
    composite_length = "DEKAD"
    level = "level_1"
    sources = pywapor.general.levels.pre_et_look_levels(level)

    print("Source of NDVI for Level 1:", sources["ndvi"])

    # Bounding box of AOI
    latlim = [-34, -33]  # first value refers to the southern border
    lonlim = [19, 20]  # first value refers to the western border

    create_ds = pywapor.pre_et_look.main(project_folder, latlim, lonlim, timelim, bin_length=composite_length)

    return create_ds


def read_et_look_input(fh):
    read_ds = xr.open_dataset(fh, decode_coords = "all")
    return read_ds


if __name__ == '__main__':
    print("Using gdal version: ", gdal.__version__)
    print("Using pywapor version: ", pywapor.__version__)

    # ds = create_et_look_input()
    # Set the path to et_look_in.nc file
    fh = ""
    ds = read_et_look_input(fh)

    # Access the coordinate reference system and boundaries
    print("CRS", ds.rio.crs)
    print("Bounds", ds.rio.bounds())
    print("Resolution", ds.rio.resolution())

    ds.z.plot()
    plt.show()

    ds.z.plot()
    plt.show()

    ds.ndvi.isel(time_bins = 0).plot()
    plt.show()

    ds_out = pywapor.et_look.main(ds)

    # Plot the daily evapotranspiration in mm
    ds_out.et_24_mm.isel(time_bins=0).plot()
    plt.show()






