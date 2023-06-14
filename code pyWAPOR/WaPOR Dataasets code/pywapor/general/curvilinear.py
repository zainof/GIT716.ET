import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, cKDTree
from dask.diagnostics import ProgressBar

def create_grid(input_ds, dx, dy, bb = None, precision = 0.1):
    """Creates a rectolinear grid with specified pixel size.

    Parameters
    ----------
    input_ds : xr.Dataset
        Dataset whose `x` and `y` dimensions will be used to create the grid.
    dx : float
        Pixel size in x-direction.
    dy : float
        Pixel size in y-direction.
    bb : list, optional
        Boundingbox for the grid (`[xmin, ymin, xmax, ymax]`), by default None.
    precision : float, optional
        Which decimal to use to snap the topleft corner of the grid, by default 0.1.

    Returns
    -------
    xr.Dataset
        Example dataset with `x` and `y` coordinates.
    """

    # precision: The left and lower bounds will snap to a `1/precision`-th degree 
    # below smallest coordinate in input_ds.

    # Open input dataset if necessary.
    if isinstance(input_ds, str):
        input_ds = xr.open_dataset(input_ds, chunks = "auto")

    if isinstance(bb, type(None)):
        bb = [input_ds.x.min().values, input_ds.y.min().values, input_ds.x.max().values, input_ds.y.max().values]

    # Determine box that is just larger then domain in input_ds.
    xmin = np.floor(bb[0] * precision**-1) / precision**-1
    xmax = np.ceil(bb[2] * precision**-1) / precision**-1
    ymin = np.floor(bb[1] * precision**-1) / precision**-1
    ymax = np.ceil(bb[3] * precision**-1) / precision**-1

    # Determine coordinates.
    gridx = np.arange(xmin, xmax + dx, dx)
    gridy = np.arange(ymin, ymax + dy, dy)

    # Wrap coordinates in xr.Dataset.
    grid_ds = xr.Dataset(None, coords = {"y": gridy, "x": gridx})
    
    # Add relevant variables.
    for var in input_ds.data_vars:
        dims = [(x, input_ds[x]) for x in input_ds[var].dims if x not in input_ds.y.dims]
        dims += [("y", grid_ds.y), ("x", grid_ds.x)]
        grid_ds[var] = xr.DataArray(coords = {k:v for k,v, in dims}, dims = [x[0] for x in dims])

    return grid_ds

def regrid(grid_ds, input_ds, max_px_dist = 10):
    """Convert a curvilinear grid to a rectolinear grid.

    Parameters
    ----------
    grid_ds : xr.Dataset
        Dataset with a rectolinear grid.
    input_ds : xr.Dataset
        Dataset with variables defined on a curvilinear grid.
    max_px_dist : int, optional
        Controls how far data gaps are (spatially) interpolated, by default 10.

    Returns
    -------
    xr.Dataset
        Dataset with variables on a rectolinear grid.
    """

    # Create output dataset
    output_ds = grid_ds.stack({"grid_pixel": ("y", "x")})
    no_pixel = output_ds.grid_pixel.size

    # Create intermediate dataset without multi-index
    grid_adj_ds = output_ds.assign_coords({"i": ("grid_pixel", range(no_pixel))}).set_index(grid_pixel="i")
    grid_adj_ds = grid_adj_ds.drop_vars(["x", "y"]).assign({"x": ("grid_pixel", grid_adj_ds.x.values),
                                                       "y": ("grid_pixel", grid_adj_ds.y.values)})

    # Determine bounding box of current chunk.
    bb = [grid_adj_ds.x.min(), grid_adj_ds.y.min(), grid_adj_ds.x.max(), grid_adj_ds.y.max()]

    # Determine pixel size.
    dx = np.median(np.unique(np.diff(grid_ds.x)))
    dy = np.median(np.unique(np.diff(grid_ds.y)))

    # Open input dataset if necessary.
    if isinstance(input_ds, str):
        input_ds = xr.open_dataset(input_ds, chunks = "auto")

    # Move `x` and `y` to variables, so they can be masked later on.
    input_ds = input_ds.reset_coords(["x", "y"])

    # Filter out irrelevant input data for current chunk.
    mask = ((input_ds.y >= bb[1] - dy) & 
            (input_ds.y <= bb[3] + dy) & 
            (input_ds.x >= bb[0] - dx) & 
            (input_ds.x <= bb[2] + dx))

    # Mask irrelevant pixels.
    data = input_ds.where(mask, drop = False)

    # Check if there is enough input data for current chunk.
    if np.any(np.array([data[x].count().values for x in data.data_vars if x not in ["x", "y"]]) < 10):
        return output_ds.unstack()

    # --heavy-> Transform input data from 2D to 1D and remove empty pixels.
    data_pixel = data.stack({"pixel": input_ds.x.dims}).dropna("pixel")

    # Load input data coordinates for scipy.
    xy = np.dstack((data_pixel.x.values,
                    data_pixel.y.values))[0]

    # Determine distances between grid pixel and nearest input data point.
    tree = cKDTree(xy)
    xi = np.stack(np.meshgrid(grid_ds.x, grid_ds.y), axis = 2)
    dists = xr.DataArray(tree.query(xi)[0], dims = ("y", "x"), coords = { "y": grid_ds.y, "x": grid_ds.x})
    dists_pixel = dists.stack({"grid_pixel": ("y", "x")}).drop("grid_pixel")

    # Load pixel coordinates for scipy functions, excluding pixels 
    # for which input data is too far away.
    max_dist = np.min([max_px_dist * dx, max_px_dist * dy])
    grid_adj_ds = grid_adj_ds.where(dists_pixel < max_dist, drop = True)
    uv = np.dstack((grid_adj_ds.x.values, grid_adj_ds.y.values))[0]

    # 2D Delaunay tessellation.
    # Also see: https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids
    tri = Delaunay(xy)

    # Find simplex for grid points.
    simplex = tri.find_simplex(uv)

    # Determine vertices and weights.
    vtx = np.take(tri.simplices, simplex, axis = 0)
    temp = np.take(tri.transform, simplex, axis = 0)
    delta = uv - temp[:, 2]
    bary = np.einsum('njk,nk->nj', temp[:, :2, :], delta)
    wts = np.hstack((bary, 1 - bary.sum(axis = 1, keepdims = True)))

    # Wrap vertices and weights in xr.DataArray.
    kwargs = {"dims": ("grid_pixel", "j"), 
              "coords": {"j": range(3), "grid_pixel": grid_adj_ds.grid_pixel}}
    wts_da = xr.DataArray(wts, **kwargs)
    vtx_da = xr.DataArray(vtx, **kwargs)

    # --heavy-> Select relevant input data.
    vls = data_pixel.drop_vars(["x", "y"]).isel(pixel = vtx_da)

    # Apply weights to input data.
    for var in vls.data_vars:
        da = xr.dot(vls[var], wts_da, dims = "j").where((wts_da > 0).all("j"))
        output_ds[var] = da.reindex(grid_pixel = dists_pixel.grid_pixel).drop_vars("grid_pixel")

    return output_ds.unstack()

if __name__ == "__main__":

    ...
    # Small test.
    # input_ds = xr.tutorial.open_dataset("rasm")
    # input_ds = input_ds.rename_dims({"x": "nx", "y": "ny"}).rename_vars({"xc":"x", "yc": "y"})

    # grid_ds = create_grid(input_ds, 1.0, 1.0).chunk({"x":500, "y":500})

    # with ProgressBar():
    #     out = xr.map_blocks(regrid, grid_ds, (input_ds,), template = grid_ds).compute()

    # from cartopy import crs as ccrs

    # fig = plt.figure(figsize=(15, 10))
    # ax1 = plt.subplot(2, 1, 1, projection = ccrs.PlateCarree())
    # ax2 = plt.subplot(2, 1, 2, projection = ccrs.PlateCarree())
    # kwargs = {"x": "x", "y": "y", "vmin": -20, "vmax": 20}
    # bla = input_ds.isel(time=0).Tair.plot.pcolormesh(ax = ax1, **kwargs)
    # ax1.coastlines()
    # ax1.set_facecolor("lightgray")
    # ax1.gridlines(draw_labels = True, color='gray', linestyle=':')
    # ax1.set_title("curvilinear")
    # out.isel(time=0).Tair.plot.pcolormesh(ax = ax2, **kwargs)
    # ax2.coastlines()
    # ax2.set_facecolor("lightgray")
    # ax2.gridlines(draw_labels = True, color='gray', linestyle=':')
    # ax2.set_title("rectolinear")
