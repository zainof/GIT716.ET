"""
Functions to create an interactive network graph based on metadata
attributes in the xr.Dataset output by `pywapor.et_look`.
"""

from pyvis.network import Network
import json

def make_node(ds, var):
    """Takes available metadata from a `var` in `ds` and formats it for use
    with pyvis.network.Network.add_node().

    Parameters
    ----------
    ds : xr.Dataset
        Dataset from which to take `var`.
    var : str
        Variable for which to prepare the network node.

    Returns
    -------
    dict
        Information required to create a network node.
    """

    if var in list(ds.variables):
        if "units" in ds[var].attrs.keys():
            units = ds[var].units
        else:
            units = ""

        if "long_name" in ds[var].attrs.keys():
            long_name = ds[var].long_name
        else:
            long_name = ""

        if "et_look_module" in ds[var].attrs.keys():
            group = ds[var].et_look_module
            shape = "dot"
            module = ds[var].et_look_module
        else:
            if len(ds[var].coords) == 0:
                group = "scalar"
            elif len(ds[var].coords) == 2:
                group = "temporal-constant"
            elif len(ds[var].coords) == 3:
                group = "temporal-input"
            elif len(ds[var].coords) > 3:
                group = "main-input"
            else:
                group = "unknown"
            shape = "square"
            module = "pre_et_look"

        if "calculated_with" in ds[var].attrs.keys():

            upstreams = ds[var].calculated_with
            if not isinstance(upstreams, list):
                upstreams = [upstreams]

            f = " = f(" + ", ".join(upstreams) + ")"
        else:
            f = ""


    else:
        long_name = ""
        units = ""
        module = ""
        group = ""
        module = ""
        f = ""
        shape = "dot"
        
    title = f"""
    <u>{long_name}</u><br><br>
    <b>{var}</b>{f} <br><br>
    Units: [{units}] <br>
    Module: {module}
    """

    return {"n_id": var, "label": var, "title": title, "group": group, "shape": shape}

def create_network(ds, fh, exclude = ["scalar", "temporal-input"]):
    """Creates a network graph for a dataset output by et_look.

    Parameters
    ----------
    ds : xr.Dataset
        Output of `et_look`.
    fh : str
        Path to where to save the network graph. Extension should be `.html`.
    exclude : list, optional
        Which types of variables to exclude from the graph. `scalar` excludes
        variables that have neither a spatial nor temporal dimension. `temporal-input`
        excludes variables that have only a temporal dimension. `temporal-constant`
        excludes variables that have only spatial dimensions. By default ["scalar", "temporal-input"].
    """
    net = Network(width = "100%", height = "100%", layout = False, 
                    bgcolor='#E9ECF5', font_color='black', directed = True)

    options = {
        "physics": {
            "hierarchicalRepulsion": {
                "centralGravity": 0,
            },
        "minVelocity": 0.75,
        "solver": "hierarchicalRepulsion",
        },
        "edges": {
            "color": {
                "color": "#B5BFDE",
                "highlight": "black",
            },
            "selectionWidth": 4,
            "smooth": {
                "type": "cubicBezier",
            }
        },
        "layout": {
            "randomSeed": 460885,
            "improvedLayout": True,
        },
        "groups": network_colors(),
        "interaction": {
            "hover": True,
            "tooltipDelay": 10,
            # "zoomSpeed": 3, doesnt work.
            "navigationButtons": False,
        }
    }

    net.set_options(json.dumps(options))

    for var in list(ds.variables):

        if "calculated_with" in ds[var].attrs.keys():

            upstreams = ds[var].calculated_with
            if not isinstance(upstreams, list):
                upstreams = [upstreams]

            if not var in net.get_nodes():
                kwargs = make_node(ds, var)
                if kwargs["group"] in exclude:
                    continue
                net.add_node(**kwargs)

            for upstream in upstreams:
                if not upstream in net.get_nodes():
                    kwargs = make_node(ds, upstream)
                    if kwargs["group"] in exclude:
                        continue
                    net.add_node(**kwargs)

                net.add_edge(upstream, var)

    net.show(fh)

def network_colors():
    """Define default styling for network graph.

    Returns
    -------
    dict
        Default styles for nodes and connections in network graph.
    """
    group_configs = {
            "main-input": {
                "color": {
                "background": "#122332",
                "border": "#14293B",
                "highlight": {
                    "border": "#2A4257",
                    "background": "#183751"
                },
                "hover": {
                    "background": "#2A4257",
                    "border": "#183751"
                }
                },
                "borderWidth": 1,
                "borderWidthSelected": 2
            },
            "pywapor.et_look_v2.meteo": {
                "color": {
                "border": "#2B7CE9",
                "background": "#97C2FC",
                "highlight": {
                    "border": "#2B7CE9",
                    "background": "#D2E5FF"
                },
                "hover": {
                    "border": "#2B7CE9",
                    "background": "#D2E5FF"
                }
                }
            },
            "pywapor.et_look_v2.radiation": {
                "color": {
                "border": "#FFA500",
                "background": "#FFFF00",
                "highlight": {
                    "border": "#FFA500",
                    "background": "#FFFFA3"
                },
                "hover": {
                    "border": "#FFA500",
                    "background": "#FFFFA3"
                }
                }
            },
            "pywapor.et_look_v2.solar_radiation": {
                "color": {
                "border": "#FA0A10",
                "background": "#FB7E81",
                "highlight": {
                    "border": "#FA0A10",
                    "background": "#FFAFB1"
                },
                "hover": {
                    "border": "#FA0A10",
                    "background": "#FFAFB1"
                }
                }
            },
            "pywapor.et_look_v2.roughness": {
                "color": {
                "border": "#41A906",
                "background": "#7BE141",
                "highlight": {
                    "border": "#41A906",
                    "background": "#A1EC76"
                },
                "hover": {
                    "border": "#41A906",
                    "background": "#A1EC76"
                }
                }
            },
            "pywapor.et_look_v2.leaf": {
                "color": {
                "border": "#E129F0",
                "background": "#EB7DF4",
                "highlight": {
                    "border": "#E129F0",
                    "background": "#F0B3F5"
                },
                "hover": {
                    "border": "#E129F0",
                    "background": "#F0B3F5"
                }
                }
            },
            "pywapor.et_look_v2.unstable": {
                "color": {
                "border": "#7C29F0",
                "background": "#AD85E4",
                "highlight": {
                    "border": "#7C29F0",
                    "background": "#D3BDF0"
                },
                "hover": {
                    "border": "#7C29F0",
                    "background": "#D3BDF0"
                }
                }
            },
            "pywapor.et_look_v2.resistance": {
                "color": {
                "border": "#C37F00",
                "background": "#FFA807",
                "highlight": {
                    "border": "#C37F00",
                    "background": "#FFCA66"
                },
                "hover": {
                    "border": "#C37F00",
                    "background": "#FFCA66"
                }
                }
            },
            "pywapor.et_look_v2.neutral": {
                "color": {
                "border": "#4220FB",
                "background": "#6E6EFD",
                "highlight": {
                    "border": "#4220FB",
                    "background": "#9B9BFD"
                },
                "hover": {
                    "border": "#4220FB",
                    "background": "#9B9BFD"
                }
                }
            },
            "pywapor.et_look_v2.evapotranspiration": {
                "color": {
                "border": "#FD5A77",
                "background": "#FFC0CB",
                "highlight": {
                    "border": "#FD5A77",
                    "background": "#FFD1D9"
                },
                "hover": {
                    "border": "#FD5A77",
                    "background": "#FFD1D9"
                }
                }
            },
            "temporal-constant": {
                "color": {
                "border": "#4AD63A",
                "background": "#C2FABC",
                "highlight": {
                    "border": "#4AD63A",
                    "background": "#E6FFE3"
                },
                "hover": {
                    "border": "#4AD63A",
                    "background": "#E6FFE3"
                }
                }
            },
            "pywapor.et_look_dev.biomass": {
                "color": {
                "border": "#990000",
                "background": "#EE0000",
                "highlight": {
                    "border": "#BB0000",
                    "background": "#FF3333"
                },
                "hover": {
                    "border": "#BB0000",
                    "background": "#FF3333"
                }
                }
            },
            "pywapor.et_look_dev.leaf": {
                "color": {
                "border": "#FF6000",
                "background": "#FF6000",
                "highlight": {
                    "border": "#FF6000",
                    "background": "#FF6000"
                },
                "hover": {
                    "border": "#FF6000",
                    "background": "#FF6000"
                }
                }
            },
            "pywapor.et_look_v2.stress": {
                "color": {
                "border": "#97C2FC",
                "background": "#2B7CE9",
                "highlight": {
                    "border": "#D2E5FF",
                    "background": "#2B7CE9"
                },
                "hover": {
                    "border": "#D2E5FF",
                    "background": "#2B7CE9"
                }
                }
            }
            }
    return group_configs

if __name__ == "__main__":

    import xarray as xr

    fh = r"/Users/hmcoerver/Downloads/pywapor_v2_network.html"

    ds = xr.open_dataset(r"/Users/hmcoerver/pywapor_notebooks/level_1/et_look_output___.nc")

    create_network(ds, fh)

    print("done")