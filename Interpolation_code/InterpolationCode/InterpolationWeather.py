import arcpy
from arcpy import env  
from arcpy.sa import *
import pandas as pd

input_gdb = "WeatherData/Data Wrangled MeteoCSV.csv" #Input of the weatherdata
output_feature_class = "weather_points.shp" #output name of the shapefile created from XYTableToPoint
x_field = "longitude"
y_field = "latitude"

env.workspace = "C:/Users/jurge/Desktop/Stellenbosch/2023(Hons)/Modules/GIT 716/Hackathon/InterpolationCode/WeatherDATAshapefile" #Filepath to where the shapefile that is created from the XYTableToPoint function is saved, apologies only the absolute filepath allows the script to run
arcpy.env.overwriteOutput = True  # Enable overwriting
arcpy.env.extent = "WeatherData/250m_Extent_WGS84.tif" #Will ensure the extend is the same as the input raster
arcpy.env.snapRaster = "WeatherData/250m_Extent_WGS84.tif" #Will ensure the rasters overlay with eachother
cellsize = "WeatherData/250m_Extent_WGS84.tif" #set the cellsize as the cell size of the input raster


# Convert CSV to shapefile using XYTableToPoint
x_field = "longitude"
y_field = "latitude"
savelocation = "InterpolateOut/"


output_csv = "C:/Users/jurge/Desktop/Stellenbosch/2023(Hons)/Modules/GIT 716/Hackathon/InterpolationCode/WeatherData/FilteredData.csv" #Filepath to where the filtered csv is saved, apologies only the absolute filepath allows the script to run
date_to_select = '2023/01/01' #Set the date to the specific day you want to create an interpolation for

# Read the input CSV into a pandas DataFrame
df = pd.read_csv(input_gdb)

# Filter the DataFrame based on the desired date
filtered_df = df[df['date'] == date_to_select]

# Save the filtered data to a new CSV file
filtered_df.to_csv(output_csv, index=False)


arcpy.management.XYTableToPoint(output_csv, output_feature_class, x_field, y_field) #Creates the weather shapefile that will be interpolated

# Interpolate windspeed using IDW
outIDWwindspeed = Idw(output_feature_class, "wind_speed", cellsize, 2, RadiusVariable(12))
outIDWwindspeed.save(f"{savelocation}2023-01-01_windspeed.tif")
    
outIDWperc = Idw(output_feature_class, "rain_sum", cellsize, 2, RadiusVariable(12))
outIDWperc.save(f"{savelocation}2023-01-01_percipitation.tif")
    
outIDWhumd = Idw(output_feature_class, "humidity_m", cellsize, 2, RadiusVariable(12))
outIDWhumd.save(f"{savelocation}2023-01-01_humidity.tif")
    
outIDWrad = Idw(output_feature_class, "solar_rad_", cellsize, 2, RadiusVariable(12))
outIDWrad.save(f"{savelocation}2023-01-01_solar_radiation.tif")