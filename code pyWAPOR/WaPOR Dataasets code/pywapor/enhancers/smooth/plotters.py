import matplotlib.pyplot as plt
import os
import string
import warnings
import numpy as np
from matplotlib import ticker, colors
import glob
import xarray as xr
import tqdm
import glob
from joblib import Parallel, delayed

def plot_point(ax, point_ds, var, ylim = [-0.2, 1], t_idx = None, title = True, xtick = True):
    cmap = plt.cm.get_cmap('tab10')
    handles = []
    if "sensor" in point_ds.data_vars:
        for i, sensor_name in enumerate(np.unique(point_ds["sensor"].values)):
            X = point_ds.time[point_ds.sensor == sensor_name]
            Y = point_ds[var][point_ds.sensor == sensor_name]
            if Y.count("time").values > 0:
                obj = ax.scatter(
                        X, Y,
                        color = cmap(i),
                        label = point_ds.sensor.attrs[str(int(sensor_name))],
                        )
                handles.append(obj)
    else:
        X = point_ds.time
        Y = point_ds[var]
        i = 0 
        if Y.count("time").values > 0:
            obj = ax.scatter(X, Y, color = cmap(i), label = "measurements")
            handles.append(obj)
    obj = ax.plot(point_ds["time"], point_ds[f"{var}_smoothed"], label = "smoothed", color = cmap(i + 1), marker = "X")
    if not isinstance(ylim, type(None)):
        ax.set_ylim(ylim)
    if not isinstance(t_idx, type(None)):
        ax.plot([point_ds["time"][t_idx].values] * 2, ylim, color = "black", linewidth = 5, alpha = 0.3)
    handles.append(obj[0])
    ax.grid()
    ax.set_ylabel(f"{var}")
    ax.set_facecolor("lightgray")
    if not xtick:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha="right")
        ax.set_xlabel("time [date]")
    if title:
        title_parts = []
        if "pname" in point_ds.attrs.keys():
            name = getattr(point_ds, "pname", "")
            title_parts.append(r"$\bf{" + name + r"}$")
        if "lmbda_sel" in point_ds.data_vars:
            lmbda = point_ds['lmbda_sel']
            title_parts.append(f"Lambda: {lmbda.values:.0E}")
        if "cves" in point_ds.data_vars and "lmbda_sel" in point_ds.data_vars:
            if "lmbda" in point_ds["cves"].dims:
                cve = point_ds['cves'].sel(lmbda = point_ds['lmbda_sel'], method = "nearest").values
            else:
                cve = point_ds["cves"].values
            title_parts.append(f"CVE: {cve:.4f}")
        if "a" in point_ds[f"{var}_smoothed"].attrs.keys():
            title_parts.append(f"a: {point_ds[f'{var}_smoothed'].attrs['a']}")
        if "x" in point_ds.coords:
            lat = float(point_ds.y.values)
            title_parts.append(f"Lat.: {lat:.3f}")
        if "y" in point_ds.coords:
            lon = float(point_ds.x.values)
            title_parts.append(f"Lon.: {lon:.3f}")
        full_title = ", ".join(title_parts)
        ax.set_title(full_title)
    ax.legend(handles = handles, ncols = 4, loc = "lower center")
    return handles

def plot_map(ax, da, points = None, cmap = "RdBu_r", ylim = [-1.0, 1.0], ytick = True, xtick = True, norm = None):
    assert da.ndim == 2
    im = ax.pcolormesh(da.x, da.y, da, cmap = cmap, vmin = ylim[0], vmax = ylim[1], norm=norm)
    ax.grid()
    ax.set_facecolor("lightgray")
    ax.set_title(da.name)
    if not isinstance(points, type(None)):
        ax.scatter(points[0], points[1], marker = "o", s = 100, edgecolors="black", linewidths = 2, c = [(1,1,1,0)], zorder = 100)
        for x,y,name in zip(*points):
            ax.annotate(name, (x,y), (30,0), textcoords = 'offset pixels', fontsize = 15)
    if not ytick:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel("Lat. [DD]")
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3f}"))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    if not xtick:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Lon. [DD]")
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3f}"))
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))

    return im

def plot_video_frame(ds, points, var, t_idx, folder):

    if not os.path.exists(folder):
        os.makedirs(folder)

    panels = [["A", "B"]]
    for i in range(len(points[0])):
        panels.append([str(i), str(i)])

    fig, axes = plt.subplot_mosaic(panels, figsize = (10, (len(points[0]) + 1)*4), dpi = 16**2)

    im1 = plot_map(axes["A"], ds["ndvi"].isel(time=t_idx), points=points)
    im2 = plot_map(axes["B"], ds["ndvi_smoothed"].isel(time=t_idx), ytick = False, points = points)

    for i, (lon, lat, name) in enumerate(zip(*points)):
        xtick = False if i+1 < len(points[0]) else True
        point_ds = ds.sel({"x": lon, "y": lat}, method = "nearest")
        point_ds = point_ds.assign_attrs({"pname": name})
        _ = plot_point(axes[str(i)], point_ds, var, t_idx = t_idx, title = True, xtick = xtick)

    fig.colorbar(im1)
    fig.colorbar(im2)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=0.91, wspace=None, hspace=0.3)
    fig.suptitle(np.datetime_as_string(ds.time[t_idx], unit='m'))
    fig.savefig(os.path.join(folder, f"{t_idx:>06}.png"))
    plt.close(fig)

def make_overview(ds, var, plot_folder, points = None, point_method = "equally_spaced", n = 3, offset = 0.1, **kwargs):

    if isinstance(points, type(None)):
        if point_method == "equally_spaced":
            points = create_points(ds, n = n, offset = offset)
        elif point_method == "worst":
            points = create_worst_points(ds, f"{var}_smoothed", "time", n = n)

    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    npanels = np.sum(np.isin(["lmbda_sel", var, "cves"], ds.data_vars))
    panels = [list(string.ascii_uppercase[:npanels])]
    for i in range(len(points[0])):
        panels.append([str(i)] * len(panels[0]))

    fig, axes = plt.subplot_mosaic(panels, figsize = (12, (len(points[0]) + 1)*4), dpi = 16**2)

    if var in ds.data_vars:
        ds["counts"] = ds[var].count(dim = "time")
        im1 = plot_map(axes[panels[0].pop(0)], ds["counts"], ylim = [None, None], cmap = "viridis", points = points)
        fig.colorbar(im1, label = "No. measurements [-]")

    if np.all(np.isin(["cves", "lmbda_sel"], ds.data_vars)):
        if "lmbda" in ds["cves"].dims:
            ds["cve"] = ds["cves"].sel(lmbda = ds["lmbda_sel"], method = "nearest")
        else:
            ds["cve"] = ds["cves"]
        im2 = plot_map(axes[panels[0].pop(0)], ds["cve"], ylim = [0, np.nanpercentile(ds["cve"], 95)], cmap = "viridis", ytick = False, points = points)
        fig.colorbar(im2, label = "Cross-val. Standard Error [-]")
    
    if "lmbda_sel" in ds.data_vars:
        if ds["lmbda_sel"].ndim == 0:
            ds["lmbda_sel"] = xr.ones_like(ds["counts"]) * ds["lmbda_sel"]
        im3 = plot_map(axes[panels[0].pop(0)], ds["lmbda_sel"], ylim = [None, None], cmap = "viridis", ytick = False, norm = colors.LogNorm(), points = points)
        fig.colorbar(im3, label = "Lambda [-]")

    for i, (lon, lat, name) in enumerate(zip(*points)):
        xtick = False if i+1 < len(points[0]) else True
        point_ds = ds.sel({"x": lon, "y": lat}, method = "nearest")
        point_ds = point_ds.assign_attrs({"pname": name})
        _ = plot_point(axes[str(i)], point_ds, var, ylim = None, t_idx = None, title = True, xtick = xtick)

    fig.subplots_adjust(left=None, bottom=None, right=None, top=0.95, wspace=None, hspace=0.3)
    fig.savefig(os.path.join(plot_folder, f"{var}_overview.png"))
    plt.close(fig)

def make_video(ds, points, fh, var, xdim = "time", new_x = None, parallel = True):

    folder = os.path.split(fh)[0]

    if not os.path.exists(folder):
        os.makedirs(folder)

    idxs = np.arange(0, ds[xdim].size, 1)
    if not isinstance(new_x, type(None)):
        idxs = idxs[~np.isin(ds[xdim], new_x)]

    if not parallel:
        for i in tqdm.tqdm(idxs):
            plot_video_frame(ds, points, var, i, folder)
    else:
        _ = Parallel(n_jobs=4)(delayed(plot_video_frame)(ds, points, var, i, folder) for i in idxs)
    
    files = np.sort(glob.glob(os.path.join(folder, "[0-9]" * 6 + ".png")))
    
    create_video(files, fh)

def create_points(ds, n = 3, offset = 0.1):
    
    if isinstance(offset, float):
        offsetx = np.floor(ds.x.size * offset).astype(int)
        offsety = np.floor(ds.y.size * offset).astype(int)
    elif isinstance(offset, int):
        offsetx = offset.copy()
        offsety = offset.copy()

    offsetx = np.min([offsetx, np.floor((ds.x.size - n) / 2).astype(int)])
    offsety = np.min([offsety, np.floor((ds.y.size - n) / 2).astype(int)])

    lons_idxs = np.linspace(0+offsetx, ds.x.size-(1+offsetx), n, dtype=int)
    lats_idxs = np.linspace(0+offsety, ds.y.size-(1+offsety), n, dtype=int)

    lons = ds.x.isel(x=lons_idxs).values
    lats = ds.y.isel(y=lats_idxs).values
    
    ys, xs = np.meshgrid(lats, lons)
    
    ys = ys.flatten().tolist()
    xs = xs.flatten().tolist()
    names = [f"P{i:>02}" for i in range(1, len(xs)+1)]
    return (xs, ys, names)

def create_worst_points(ds, var, dim, n = 8):

    ds = np.abs(ds).max(dim = dim)

    stacked_ds = ds.stack({"point": ds[var].dims})[var]

    selected = stacked_ds.isel(point = stacked_ds.argsort().values).isel(point = slice(-n-1, -1))

    points = (selected.x.values, selected.y.values, [f"P{i:>02}" for i in range(1, n+1)])
    return points

def create_video(files, video_fh, fps = 4, remove_files = True):

    try:
        import imageio.v2 as imageio
    except ImportError:
        print("--> Install `imageio` to automatically create a video")
        return

    try:
        with imageio.get_writer(video_fh, fps = fps) as writer:
            for im in files:
                writer.append_data(imageio.imread(im))
        if remove_files:
            for fn in files:
                try:
                    os.remove(fn)
                except PermissionError:
                    continue
    except ValueError as e:
        msg = getattr(e, "args", ["None"])
        if "Based on the extension, the following" in msg[0]:
            fn, ext = os.path.splitext(video_fh)
            if ext == ".mp4":
                print("--> Creating `.gif` file, install `imageio_ffmpeg` to create `.mp4`.")
                create_video(files, fn + ".gif", fps = fps)
            else:
                print("--> Unable to create video with requested extension.")
                print(e)
        else:
            raise e
