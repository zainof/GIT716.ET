# GIT716.ET
This repository details the code for modelling ET in the model of ET look. This repository was created by the GIT 716 Honors class of 2023

# Tech Stack 
- Visual Studio Code or Pycharm
- GitHub (Version control) 

# Create your python environment using Conda
- Create a new CodeSpace in order to interact with the provided code.

  <img width="323" alt="Code" src="https://github.com/zainof/GIT716.ET/assets/130638108/d7129428-b54c-4ec9-9507-aba8720d0a7b">

- Shift + Ctrl + P to open the command pallet
- Select Terminal
- If prompted to first create a Python interpreter, select the base Python 3.10.8 interpreter. 
- In the terminal, you want to insert the following code to create a new conda environment:

  conda create -n pywapor --yes -c conda-forge python pip gdal pydap numpy pandas requests matplotlib pyproj scipy pycurl pyshp joblib bs4 rasterio xarray bottleneck geojson tqdm dask rioxarray pyvis shapely lxml cachetools cdsapi sentinelsat geopy numba ipywidgets scikit-learn.
  
  <img width="695" alt="NewEnv" src="https://github.com/zainof/GIT716.ET/assets/130638108/b04dd8e7-cc62-44fa-85ba-7aadf32aef29">

- This will create a new conda environment that contains the important GDAL package.

# Installing pyWapor 
- To use your new Conda environment, input the following code into your terminal:
  
  <img width="509" alt="image" src="https://github.com/zainof/GIT716.ET/assets/130638108/8b359fcd-80d7-4353-b76b-6b1c758ea9e3">

- If conda presents an issue that the environment has not yet been initialised, just enter the following code into the terminal, kill the terminal and retry the previous step:
  
  <img width="438" alt="image" src="https://github.com/zainof/GIT716.ET/assets/130638108/82a2113a-2551-4307-a6e1-e6b2f8b46c00">

- Once in the new conda environment, the following code can be used to install pywapor: 
  
  <img width="521" alt="InstallingPywapor" src="https://github.com/zainof/GIT716.ET/assets/130638108/592396a4-ffb2-431f-a327-405abd3dcc48">

- Pywapor should now be installed. 

# Notebook Execution 
- To access the workbook and interact with the code, select the conda environment you created under the choices of interpreter:
    
  <img width="703" alt="env" src="https://github.com/zainof/GIT716.ET/assets/130638108/5ad91a0d-ebb1-4a64-a6aa-02c59b3f529f">
 
  <img width="459" alt="SelectingEnv" src="https://github.com/zainof/GIT716.ET/assets/130638108/9344f83f-b3db-4f27-8c98-1d177244a195">
 
- The notebook will now be accessible for use. 

# Terms of Use
- Before editing and changing the notebook, please create a new branch with your name to track changes. 
- Editors will then be able to push and commit changes where neccessary.
- Otherwise, contact the editors at their GITHUB email for any neccessary input required with the code. 
