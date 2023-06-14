# temperature conversions
zero_celcius = 273.15     # 0 degrees C in K

# reference values
t_ref = 293.15            # reference temperature 20 degrees celcius
p_ref = 1013.25           # reference pressure in mbar
z_ref = 0                 # sea level m

lapse = -0.0065           # lapse rate K m-1
g = 9.807                 # gravity m s-2
gc_spec = 287.0           # gas constant J kg-1 K-1
gc_dry = 2.87             # dry air gas constant mbar K-1 m3 kg-1
gc_moist = 4.61           # moist air gas constant mbar K-1 m3 kg-1
r_mw = 0.622              # ratio water particles/ air particles
sh = 1004.0               # specific heat J kg-1 K-1
lh_0 = 2501000.0          # latent heat of evaporation at 0 C [J/kg]
lh_rate = -2361           # rate of latent heat vs temperature [J/kg/C]
power = (g/(-lapse*gc_spec))
k = 0.41                  # karman constant (-)
sol = 1367                # maximum solar radiation at top of atmosphere W m-2
sb = 5.67e-8              # stefan boltzmann constant
day_sec = 86400.0         # seconds in a day
year_sec = day_sec * 365  # seconds in a year

absorbed_radiation = 0.48 # biomass factor
conversion = 0.864        # conversion biomass calculation from g s-1 m-2
                          # to kg ha-1 d-1

z0_soil = 0.001           # soil roughness m

#############
#############
#############

aod550_i = 0.01 # https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD04_L2 heb niet echt een standaard product hiervan gevonden
se_top = 0.5
porosity = 0.4

# http://lawr.ucdavis.edu/classes/SSC100/probsets/pset01.html
# 6. Calculate the porosity of a soil sample that has a bulk density of 1.35 g/cm3. Assume the particle density is 2.65 g/cm3.
# Porosity = (1-(r b/r d) x 100 = (1-(1.35/2.65)) x 100 = 49%
# page 31 flow diagram

# **effective_leaf_area_index**************************************************
# constants or predefined:
nd_min = 0.1 # 0.125
nd_max = 0.8
vc_pow = 0.7
vc_min = 0
vc_max = 0.9677324224821418
lai_pow = -0.45

# **atmospheric canopy resistance***********************************************
# constants or predefined:
diffusion_slope = -1.33
diffusion_intercept = 1.15
t_opt = 25 # optimal temperature for plant growth
t_min = 0 # minimal temperature for plant growth
t_max = 50 # maximal temperature for plant growth
vpd_slope = -0.3
rcan_max = 1000000

# **net radiation canopy******************************************************
# constants or predefined:
vp_slope = 0.14
vp_offset = 0.34
lw_slope = 1.35
lw_offset = 0.35
int_max = 0.2

# **canopy resistance***********************************************************
# constants or predefined:
r0_bare = 0.42 # 0.38
r0_bare_wet = 0.20
r0_full = 0.18
tenacity = 1.0
disp_bare = 0.0
disp_full = 0.667
fraction_h_bare = 0.65
fraction_h_full = 0.95

z_obs = 2
z_b = 100
z0m_bare = 0.0005 # 0.001

# **initial canopy aerodynamic resistance***********************************************************
# constants or predefined:
ndvi_obs_min = 0.25
ndvi_obs_max = 0.75
obs_fr = 0.25

# **ETLook.unstable.initial_friction_velocity_daily***********************************************************
# constants or predefined:
c1 = 1

# **ETLook.unstable.transpiration***********************************************************
# constants or predefined:
iter_h = 3

# **ETLook.resistance.soil_resistance***********************************************************
# constants or predefined:
r_soil_pow = -2.1
r_soil_min = 800

# **ETLook.unstable.initial_sensible_heat_flux_soil_daily***********************************************************
# constants or predefined:
#porosity = 0.4 #Note: soil dependent
#se_top = 1.0 #Note should be input !
rn_slope = 0.92
rn_offset = -61.0

# **ETLook.unstable.evaporation***********************************************************
# constants or predefined:
r0_grass = 0.23