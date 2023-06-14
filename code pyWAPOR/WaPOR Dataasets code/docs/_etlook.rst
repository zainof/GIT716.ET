:bitbucket_url: https://bitbucket.org/cioapps/pywapor/src/master/

*********
ETLook
*********

The method for calculating evapotranspiration is based on the ETLook model developed by eLEAF in 2010. 

The method for calculating total biomass production is based on the C-Fix model. 

The code in the `pywapor.et_look_v2` module of this repository, containing all core physical functions used by ETLook, was written by Henk Pelgrum (eLEAF) and Rutger Kassies (eLEAF). 

The remaining modules have been developed by Bert Coerver (FAO), Tim Hessels (WaterSat), and, in the framework of the ESA-funded ET4FAO project, Radoslaw Guzinski (DHI-GRAS), Hector Nieto (Complutig) and Laust Faerch (DHI-GRAS).

.. raw:: html

   <div height="100%", width="100%">
   <iframe width="100%" height="500vh" src="et_look_v2_network.html" title="et_look_model" style="border:0px" scrolling="no"></iframe>
   </div>

.. toctree::

   et_look_rsts/clear_sky_radiation
   et_look_rsts/soil_moisture
   et_look_rsts/evapotranspiration
   et_look_rsts/leaf
   et_look_rsts/meteo
   et_look_rsts/neutral
   et_look_rsts/radiation
   et_look_rsts/resistance
   et_look_rsts/roughness
   et_look_rsts/solar_radiation
   et_look_rsts/stress
   et_look_rsts/unstable
   et_look_rsts/biomass