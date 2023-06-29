# %% [markdown]
# ### Installation:
# 
# pip install earthenginge-api
# 
# pip install folium
# 
# pip install matplotlib
# 
# pip install rasterio

# %% [markdown]
# ### Initialization

# %%
import ee
from IPython.core.display import Image as IPythonImage
import folium
import matplotlib.pyplot as plt
import rasterio
import os
import glob
import requests
import io
from datetime import timedelta, datetime
import calendar

import rasterio.warp
from rasterio import mask
from rasterio.crs import CRS
import shapely.geometry as sg
from rasterio.transform import from_origin
from rasterio.io import MemoryFile
from rasterio.features import geometry_window
from rasterio.warp import calculate_default_transform, reproject
from rasterio.plot import reshape_as_image
from rasterio.enums import Resampling
from rasterio.transform import Affine
from PIL import Image
import numpy as np

# %% [markdown]
# ### Authentication:

# %%
ee.Authenticate()
ee.Initialize()

# %% [markdown]
# Will generate a token to kennect with gcloud on GEE:
# 
# Implementing this will require a GEE account and repo setup to use

# %% [markdown]
# ##### Test Imports and Auth wokred

# %%
#Test
# Print the elevation of Mount Everest.
dem = ee.Image('USGS/SRTMGL1_003')
xy = ee.Geometry.Point([86.9250, 27.9881])
elev = dem.sample(xy, 30).first().get('elevation').getInfo()
print('Mount Everest elevation (m):', elev)

# %% [markdown]
# ### Automation in accessing
#     - local folders
#     - clipping to extent,
#     - resample
#     - save as renamed .tiff
#     - export to dependancy folder

# %%

# Specify the input folder path
root_folder = 'Data/'
#input_folder = "Data/temp_mean/"  # Update with your input folder path
target_resolution = 250

# Specify the output folder path
output_folder = "Data_automated"  # Update with your desired folder path

# Retrieve a list of input file paths from the input folder
def get_image_files(root_folder, file_extension='.tif'):
    image_files = []

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(file_extension):
                relative_path = os.path.relpath(root, root_folder)  # Get the relative path of the subfolder
                relative_path = relative_path.replace('/\\', '/')  # Convert backslashes to forward slashes
                output_subfolder = os.path.join(output_folder, relative_path)  # Create the output subfolder path
                os.makedirs(output_subfolder, exist_ok=True)  # Create the output subfolder if it doesn't exist
                image_files.append((os.path.join(root, file), output_subfolder))  # Append the file and output subfolder tuple

    return image_files

# Function to generate the output file name based on the input file name
def generate_output_filename(input_filename):
    base_name = os.path.basename(input_filename)
    filename, extension = os.path.splitext(base_name)
    output_filename = "resampled_image_" + filename + ".tif"
    return output_filename

image_files = get_image_files(root_folder)

# Process each image file and save the output in the respective subfolder
for file, output_subfolder in image_files:
    image_path = file
    # Perform the processing and save the output TIFF file in the output folder
    
    # Define the bounding box coordinates
    # Define the region of interest (Western Cape, South Africa)
    roi = [18.723629, -34.050015, 19.390929, -33.370358]
    left, bottom, right, top = roi
    
    # Specify the target coordinate reference system (CRS) for reprojection
    target_crs = 'EPSG:4326'
    target_resolution = target_resolution  # Update with your desired target resolution in meters

    ##############
    # Read Array and MetaData
    ##############
    # Function to read local raster data and return as a variable
    def read_local(image_path):
        with rasterio.open(image_path) as src:
            array = src.read()
            image_meta = src.meta
            image_meta['crs'] = str(src.crs)
            image_meta['width'] = src.width
            image_meta['height'] = src.height
            image_meta['bounds'] = src.bounds
        return array, image_meta
    # Read the local raster image
    image_array, image_meta = read_local(image_path)
    
    # Clip the image array to the ROI
    with rasterio.open(image_path) as src:
        roi_polygon = sg.Polygon([(roi[0], roi[1]), (roi[2], roi[1]), (roi[2], roi[3]), (roi[0], roi[3])])
        clipped, transform = rasterio.mask.mask(src, [roi_polygon], crop=True)
        clipped_meta = src.meta.copy()
        # Print the bounds of the original image and the clipped image
        #print("Original Image Bounds:", src.bounds)

        # Calculate the bounds of the clipped image
        clipped_height, clipped_width = clipped.shape[1:]
        clipped_bounds = rasterio.transform.array_bounds(clipped_height, clipped_width, transform)

        # Print the bounds of the clipped image
        #print("Clipped Image Bounds:", clipped_bounds)
    
    # Calculate the roi_transform
    roi_transform = transform

    ##############
    # Reproject & Resample
    ##############
    # Define the target CRS
    target_crs = CRS.from_string(target_crs)
    height = clipped_meta['height']
    width = clipped_meta['width']
    
    # Calculate the transform and dimensions of the resampled image
    resample_transform, resample_width, resample_height = calculate_default_transform(
        clipped_meta['crs'], target_crs,
        width, height,
        left, bottom, right, top,
        dst_width=270, dst_height=275  # Update the desired number of columns and rows here
    )

    # Update the metadata with the resampled transform and dimensions
    clipped_meta.update({
        'transform': resample_transform,
        'width': resample_width,
        'height': resample_height
    })

    # Create the reprojected array
    reprojected_array = np.zeros((clipped.shape[0], target_resolution, target_resolution), dtype=clipped.dtype)


    # Reproject the clipped image array
    reproject(
        source=clipped,
        destination=reprojected_array,
        src_transform=transform,
        src_crs=clipped_meta['crs'],
        dst_transform=resample_transform,
        dst_crs=target_crs,
        resampling=Resampling.nearest
    )

    # Update the metadata of the reprojected array
    reprojected_meta = clipped_meta.copy()
    reprojected_meta['transform'] = resample_transform
    reprojected_meta['width'] = resample_width
    reprojected_meta['height'] = resample_height

    # Create a new file for the resampled image
    output_filename = generate_output_filename(image_path)
    output_path = os.path.join(output_subfolder, output_filename)

    # Write the resampled image to the output file
    with rasterio.open(output_path, 'w', **reprojected_meta) as dst:
        dst.write(reprojected_array)
        print(output_filename)


# %% [markdown]
# ### Surface Albedo

# %%
def generate_surfaceAlbedo(roi, startDate, output_folder, output_filename):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Define the bands
    bands = ['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b04', 'sur_refl_b06', 'sur_refl_b07']

    # Define the coefficients for albedo calculation
    coefficients = [0.215/10000, 0.215/10000, 0.242/10000, 0.18/10000, 0.112/10000, 0.036/10000]

    # Define the number of iterations
    numIterations = 3  # Split the month into three 10-day periods

    # Initialize the counter
    counter = 1

    # Loop for the specified number of iterations
    for i in range(numIterations):
        # Calculate the start and end dates for each 10-day period
        startDay = i * 10 + 1
        endDay = (i + 1) * 10
        startDatePeriod = ee.Date(startDate).advance(startDay - 1, 'day')
        endDatePeriod = ee.Date(startDate).advance(endDay, 'day')

        # Set the start time to 00:00 and end time to 23:59 of the current period
        currentDateTimeRange = ee.DateRange(startDatePeriod, endDatePeriod.advance(-1, 'second'))

        # Import the MODIS dataset for the current date range
        dataset = ee.ImageCollection('MODIS/061/MOD09GA') \
            .filterDate(currentDateTimeRange) \
            .select(bands) \
            .map(lambda image: image.clip(roi))

        # Function to calculate surface albedo
        def calculateAlbedo(image):
            albedo = image.expression(
                'coeff1 * B1 + coeff2 * B2 + coeff3 * B3 + coeff4 * B4 + coeff5 * B6 + coeff6 * B7', {
                    'coeff1': coefficients[0],
                    'coeff2': coefficients[1],
                    'coeff3': coefficients[2],
                    'coeff4': coefficients[3],
                    'coeff5': coefficients[4],
                    'coeff6': coefficients[5],
                    'B1': image.select('sur_refl_b01'),
                    'B2': image.select('sur_refl_b02'),
                    'B3': image.select('sur_refl_b03'),
                    'B4': image.select('sur_refl_b04'),
                    'B6': image.select('sur_refl_b06'),
                    'B7': image.select('sur_refl_b07')
                }).rename('albedo')

            return image.addBands(albedo)

        # Apply the albedo calculation to each image in the collection
        albedoDataset = dataset.map(calculateAlbedo)

        # Visualize the albedo layer
        albedoVisParams = {
            'min': 0,
            'max': 5000,
            'palette': 'FFFFFF, CE7E45, DF923D, F1B555, FCD163, 99B718,'
                       '74A901, 66A000, 529400, 3E8601, 207401, 056201,'
                       '004C00, 023B01, 012E01, 011D01, 011301'
        }  # Adjust the min and max values as needed

        # Create a folium map object
        my_map = folium.Map(location=[-30, 21], zoom_start=6)

        def add_ee_layer(ee_image_object, vis_params, name):
            map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
            folium.raster_layers.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format,
                attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
                name=name,
                overlay=True,
                control=True
            ).add_to(my_map)

        # Add the clipped albedo image to the map object
        add_ee_layer(albedoDataset.select('albedo').median(), albedoVisParams, 'Surface Albedo - Period ' + str(counter))
        
        # Save the map as an HTML file
        def timeName(startDate, counter):
            start_date_str = startDate.format('YYYY-MM-DD')
            date_obj = datetime.strptime(start_date_str, '%Y-%m-%d')
            month_nr = str(date_obj.month)
            year_nr = str(date_obj.year)
            day_nr = str(date_obj.day)
            add = 1
            if counter == 1:
                day_nr = day_nr.zfill(2)
                add = 10
            elif counter == 2:
                day_nr = 11
                add = 20
            else :
                day_nr = 21
                add = calendar.monthrange(date_obj.year, date_obj.month)[1]
            name = year_nr.zfill(2)+'-'+month_nr.zfill(2)+'-'+str(day_nr)+'_to_'+str(add).zfill(2)+'_'
            return name

        _date= timeName(startDate,counter)
        
        output_html = os.path.join(output_folder, str(_date) + output_filename + '.html')
        
        my_map.save(output_html)

        image = albedoDataset.select('albedo').median()
        # Export the surface albedo as a GeoTIFF
        # Get the download URL for the GeoTIFF
        download_url = image.getDownloadURL({
            'name': output_filename,
            #'scale': 250,  # Pixel scale in meters
            'crs': 'EPSG:4326',  # Coordinate reference system
            'region': roi,  # Region of interest
            'dimensions': '270x275',
            'output_type': 'Float32',  # Set the output data type
            'format': 'GEO_TIFF'
        })

        # Send a request to download the GeoTIFF file
        response = requests.get(download_url)

        # Check if the request was successful
        if response.status_code == 200:
            # Specify the output file path
            output_tiff = os.path.join(output_folder, str(_date) + output_filename + '.tif')

            # Save the response content as a GeoTIFF file
            with open(output_tiff, 'wb') as file:
                file.write(response.content)

            # Increase the counter by 1
            counter += 1
            # Print the output TIFF file path
            print('Output TIFF file:', output_tiff)
        else:
            # Print an error message if the request failed
            print('Failed to download the GeoTIFF file.')
        

#Define the region of interest (Western Cape, South Africa)
roi = ee.Geometry.Rectangle([18.723629, -34.050015, 19.390929, -33.370358]);

#Define the initial date range
startDate = '2023-01-01';
#endDate = '2023-02-01';

# Specify the output folder and filename
output_folder = 'Data_automated/SurfaceAlbedo'
output_filename = 'surfaceAlbedo'

# Generate the TIFF file from GEE data and save it to the specified folder
output_tiff = generate_surfaceAlbedo(roi, startDate, output_folder, output_filename) #endDate


# %% [markdown]
# ### NDVI

# %%

def getDays(startDate):
    start_date_str = startDate.format('YYYY-MM-DD')
    date_obj = datetime.strptime(start_date_str, '%Y-%m-%d')
    month_nr = date_obj.month
    year_nr = date_obj.year
    days = calendar.monthrange(year_nr, month_nr)[1] 
    return days

def generate_NDVI(roi, start_date, output_folder, output_filename):
     # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    startDate = start_date
    
    
    # Extract year and month from the start date
    start_date = ee.Date(start_date)
    year = start_date.get('year')
    month = start_date.get('month')

    # Calculate the end date of the month
    last_day = start_date.advance(1, 'month').advance(-1, 'day')
    end_date = last_day.format("YYYY-MM-DD")
    
    # Determine the number of days in the month
    num_days = getDays(startDate)
    
    # Initialize a variable to keep track of the remaining days
    remaining_days = num_days

    # Iterate over each 10-day period of the month
    for i in range(1, num_days + 1, 10):
        temp_e = None
        
        # Calculate the end day of the current period
        end_day = min(i + 9, num_days)

        # Pad the day with leading zeros if necessary
        start_day = str(i).zfill(2)
        end_day = str(end_day).zfill(2)

        # Define the start and end dates for each 10-day period
        start_date_period = ee.Date.fromYMD(year, month, int(start_day))
        end_date_period = None
        remaining_days -= 10
        
        if remaining_days < 0:
            # ensure program breaks/ends if num_days = end_day
            break
        
        if remaining_days < 10:
            #To account for months with odd numbers days (27, 28, 31)
            #print('remaining ', remaining_days)
            temp_e = ee.Date.fromYMD(year, month, int(end_day)).advance(1, 'day')
            temp_endDay = int(end_day) + remaining_days
            end_day = str(temp_endDay)
        else :
            temp_e = ee.Date.fromYMD(year, month, int(end_day))
        
        end_date_period = temp_e
        
        # Filter the dataset based on the time period and region of interest
        #MODIS_006_MOD09GQ
        dataset = ee.ImageCollection('MODIS/006/MOD09GQ') \
            .filterBounds(roi) \
            .filterDate(start_date_period, end_date_period)

        # Function to calculate NDVI
        def calculate_ndvi(image):
            ndvi = image.normalizedDifference(['sur_refl_b02', 'sur_refl_b01'])
            return ndvi.rename('NDVI').copyProperties(image, ['system:time_start'])

        # Map over the image collection to calculate NDVI
        ndvi_collection = dataset.map(calculate_ndvi)

        # Calculate the maximum NDVI for the period
        ndvi_composite = ndvi_collection.max()

        # Clip the composite to the region of interest
        clipped_composite = ndvi_composite.clip(roi)

        # Visualize the NDVI layer
        VisParams = {
            'min': 0,
            'max': 1,
            'palette': ['FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718',
                        '74A901', '66A000', '529400', '3E8601', '207401', '056201',
                        '004C00', '023B01', '012E01', '011D01', '011301']
        }  # Adjust the min and max values as needed

        # Create a folium map object
        my_map = folium.Map(location=[-30, 21], zoom_start=6)

        def add_ee_layer(ee_image_object, vis_params, name):
            map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
            folium.raster_layers.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format,
                attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
                name=name,
                overlay=True,
                control=True
            ).add_to(my_map)

        # Add the clipped NDVI image to the map object
        add_ee_layer(clipped_composite.select('NDVI'), VisParams, 'NDVI')

        # Save the map as an HTML file
        def timeName(startDate):
            start_date_str = startDate.format('YYYY-MM-DD')
            date_obj = datetime.strptime(start_date_str, '%Y-%m-%d')
            month_nr = str(date_obj.month)
            year_nr = str(date_obj.year)
            return year_nr.zfill(2)+'-'+month_nr.zfill(2)+'-'

        _date= timeName(startDate)
        _day= start_day
        
        output_html = os.path.join(output_folder, str(_date) + _day + '_to_' + end_day +'_'+ output_filename + '.html')
        my_map.save(output_html)

        image = clipped_composite.select('NDVI')
        # Export the surface albedo as a GeoTIFF
        # Get the download URL for the GeoTIFF
        download_url = image.getDownloadURL({
            'name': output_filename,
            #'scale': 250,  # Pixel scale in meters
            'crs': 'EPSG:4326',  # Coordinate reference system
            'region': roi,  # Region of interest
            'dimensions': '270x275',
            'output_type': 'Float32',  # Set the output data type
            'format': 'GEO_TIFF'
        })

        # Send a request to download the GeoTIFF file
        response = requests.get(download_url)

        # Check if the request was successful
        if response.status_code == 200:
            # Specify the output file path
            output_tiff = os.path.join(output_folder, str(_date) + _day + '_to_' + end_day +'_'+ output_filename + '.tif')

            # Save the response content as a GeoTIFF file
            with open(output_tiff, 'wb') as file:
                file.write(response.content)

            # Print the output TIFF file path
            print('Output TIFF file:', output_tiff)
        else:
            # Print an error message if the request failed
            print('Failed to download the GeoTIFF file.')
        

#Define the region of interest (Western Cape, South Africa)
roi = ee.Geometry.Rectangle([18.723629, -34.050015, 19.390929, -33.370358]);

#Define the initial date range
startDate = '2023-01-01';
#endDate = '2023-02-01';

# Specify the output folder and filename
output_folder = 'Data_automated/NDVI'
output_filename = 'ndvi'

# Generate the TIFF file from GEE data and save it to the specified folder
output_tiff = generate_NDVI(roi, startDate, output_folder, output_filename) #endDate


# %% [markdown]
# ### LST

# %%
def getDays(startDate):
    start_date_str = startDate.format('YYYY-MM-DD')
    date_obj = datetime.strptime(start_date_str, '%Y-%m-%d')
    month_nr = date_obj.month
    year_nr = date_obj.year
    days = calendar.monthrange(year_nr, month_nr)[1] 
    return days

def generate_LST(roi, startDate, output_folder, output_filename):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
   
    startDate = startDate
    
    # Extract year and month from the start date
    start_date = ee.Date(startDate)
    year = start_date.get('year')
    month = start_date.get('month')

    # Calculate the end date of the month
    last_day = start_date.advance(1, 'month').advance(-1, 'day')
    end_date = last_day
    
    # Determine the number of days in the month
    num_days = getDays(startDate)
    
    # Initialize a variable to keep track of the remaining days
    remaining_days = num_days

    # Iterate over each day of the month
    for i in range(1, num_days + 1):
        
        # Format the start and end dates as strings
        start_date_str = datetime(int(year.getInfo()), int(month.getInfo()), i).strftime('%Y-%m-%d')
        
        end_date_str = datetime(int(year.getInfo()), int(month.getInfo()), i).strftime('%Y-%m-%d')
        
        # Set the start time to 00:00 and end time to 23:59 of the current day
        start_date_str += 'T00:00:00'
        end_date_str += 'T23:59:59'
        
        # Create an ImageCollection and filter by date and clip to the geometry
        dataset = ee.ImageCollection('MODIS/061/MOD11A1') \
            .filterBounds(roi) \
            .filterDate(start_date_str, end_date_str) \
            .map(lambda image: image.clip(roi))
              
        # Function to calculate LST
        def calculate_lst(image):
            lst = image.clip(roi)
            lst_day_1km = lst.select('LST_Day_1km')
            return lst_day_1km.rename('LST_Day_1km').copyProperties(image, ['system:time_start'])

        # Map over the image collection to calculate LST
        lst_collection = dataset.map(calculate_lst)

        # Select the 'LST_Day_1km' band
        # Select the first image in the collection
        landSurfaceTemperature = lst_collection.first()
        #landSurfaceTemperature = lst_collection.select('LST_Day_1km')
        
        # Create a folium map object
        my_map = folium.Map(location=[-30, 21], zoom_start=6)

        # Define visualization parameters for land surface temperature
        landSurfaceTemperatureVis = {
            'min': 13000.0,
            'max': 16500.0,
            'palette': [
                '040274', '040281', '0502a3', '0502b8', '0502ce', '0502e6',
                '0602ff', '235cb1', '307ef3', '269db1', '30c8e2', '32d3ef',
                '3be285', '3ff38f', '86e26f', '3ae237', 'b5e22e', 'd6e21f',
                'fff705', 'ffd611', 'ffb613', 'ff8b13', 'ff6e08', 'ff500d',
                'ff0000', 'de0101', 'c21301', 'a71001', '911003'
            ]
        }
        
        def add_ee_layer(ee_image_object, vis_params, name):
            map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
            folium.raster_layers.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format,
                attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
                name=name,
                overlay=True,
                control=True
            ).add_to(my_map)

        # Add the clipped NDVI image to the map object
        add_ee_layer(landSurfaceTemperature, landSurfaceTemperatureVis, 'Land Surface Temperature')

        # Save the map as an HTML file
        def timeName(startDate):
            start_date_str = startDate.format('YYYY-MM-DD')
            date_obj = datetime.strptime(start_date_str, '%Y-%m-%d')
            month_nr = str(date_obj.month)
            year_nr = str(date_obj.year)
            day_nr = str(date_obj.day)
            return year_nr.zfill(2)+'-'+month_nr.zfill(2)+'-'

        _date= timeName(startDate)
        _day = str(i).zfill(2)+'_'
        
        output_html = os.path.join(output_folder, str(_date) + _day + output_filename + '.html')
        # 2023-01-01_filename.tif 
        my_map.save(output_html)
        
        # Export the surface albedo as a GeoTIFF
        # Get the download URL for the GeoTIFF
        image = landSurfaceTemperature.multiply(0.02)
        download_url = image.getDownloadURL({
            'name': output_filename,
            #'scale': 250,  # Pixel scale in meters
            'crs': 'EPSG:4326',  # Coordinate reference system
            'region': roi,  # Region of interest
            'dimensions': '270x276',
            'output_type': 'Float32',  # Set the output data type
            'format': 'GEO_TIFF'
        })

        # Send a request to download the GeoTIFF file
        response = requests.get(download_url)

        # Check if the request was successful
        if response.status_code == 200:
            # Specify the output file path
            output_tiff = os.path.join(output_folder, str(_date) + _day + output_filename + '.tif')

            # Save the response content as a GeoTIFF file
            with open(output_tiff, 'wb') as file:
                file.write(response.content)

            # Print the output TIFF file path
            print('Output TIFF file:', output_tiff)
        else:
            # Print an error message if the request failed
            print('Failed to download the GeoTIFF file.')
    

#Define the region of interest (Western Cape, South Africa)
roi = ee.Geometry.Rectangle([18.723629, -34.050015, 19.390929, -33.370358]);

#Define the initial date range
startDate = '2023-01-01';
#endDate = '2023-02-01';

# Specify the output folder and filename
output_folder = 'Data_automated/LST'
output_filename = 'lst'

# Generate the TIFF file from GEE data and save it to the specified folder
output_tiff = generate_LST(roi, startDate, output_folder, output_filename) #endDate



