from setuptools import setup, find_packages

setup(
    name = 'pywapor',
    version = '3.3.4',
    url = 'https://www.fao.org/aquastat/py-wapor/',
    author = "FAO",
    author_email = "bert.coerver@fao.org",
    license = "Apache",
    packages = find_packages(include = ['pywapor', 'pywapor.*']),
    include_package_data=True,
    python_requires='>=3.7',
    install_requires = [
        'gdal',
        'xarray>=0.20',
        'numpy',
        'pydap',
# - NOTE in pandas == 2.0.0, the buffering of timelim (e.g. in collect.product.MODIS.download) doesnt work correctly.
# So fo now sticking to 1.5.3 and waiting for 2.x.x to become more stable.
        'pandas<2.0.0',
        'requests',
        'matplotlib',
# - NOTE otherwise opendap gives problem in colab, in conda env netcdf=1.6.0 
# works fine -> https://github.com/Unidata/netcdf4-python/issues/1179
# - (fixed) NOTE also, cant install netcdf4 with conda, because the conda-forge distribution cant open
# the PROBAV HDF files. See issues https://github.com/Unidata/netcdf4-python/issues/1182 and
# https://github.com/conda-forge/netcdf4-feedstock/issues/136
# - NOTE set libnetcdf=4.8 in conda otherwise this happend:
# https://github.com/pydata/xarray/issues/7549 (also see https://github.com/SciTools/iris/issues/5187)
        'netcdf4<1.6.0', 
        'pyproj',
        'scipy',
        'pycurl',
        'pyshp',
        'joblib',
        'bs4',
        'rasterio',
        'bottleneck>=1.3.1',
        'geojson',
        'tqdm',
        'dask',
        'rioxarray',
        'python_log_indenter',
        'cryptography',
        'pyvis',
        'cachetools',
        'cdsapi',
        'sentinelsat',
        'shapely',
        'lxml',
        'geopy',
        'scikit-learn',
        'numba',
        'xmltodict',
# - (fixed) NOTE Another fix for Colab... https://github.com/googlecolab/colabtools/issues/3134
        # 'importlib-metadata==4.13.0',
    ],
    classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)