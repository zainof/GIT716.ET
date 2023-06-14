import rasterio

def get_crss(name = None):
    crss = {"MODIS": rasterio.crs.CRS.from_wkt("""
            PROJCRS["unnamed",
                BASEGEOGCRS["Unknown datum based upon the custom spheroid",
                    DATUM["Not_specified_based_on_custom_spheroid",
                        ELLIPSOID["Custom spheroid",6371007.181,0,
                            LENGTHUNIT["metre",1,
                                ID["EPSG",9001]]]],
                    PRIMEM["Greenwich",0,
                        ANGLEUNIT["degree",0.0174532925199433,
                            ID["EPSG",9122]]]],
                CONVERSION["unnamed",
                    METHOD["Sinusoidal"],
                    PARAMETER["Longitude of natural origin",0,
                        ANGLEUNIT["degree",0.0174532925199433],
                        ID["EPSG",8802]],
                    PARAMETER["False easting",0,
                        LENGTHUNIT["Meter",1],
                        ID["EPSG",8806]],
                    PARAMETER["False northing",0,
                        LENGTHUNIT["Meter",1],
                        ID["EPSG",8807]]],
                CS[Cartesian,2],
                    AXIS["easting",east,
                        ORDER[1],
                        LENGTHUNIT["Meter",1]],
                    AXIS["northing",north,
                        ORDER[2],
                        LENGTHUNIT["Meter",1]]]
            """),

        "WGS84": rasterio.crs.CRS.from_epsg(4326),
    }
    if not isinstance(name, type(None)):
        return crss[name]
    else:
        return crss