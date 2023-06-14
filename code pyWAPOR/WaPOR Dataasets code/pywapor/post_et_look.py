import numpy as np
from osgeo import gdal
import pywapor.general.processing_functions as PF
import warnings
import os
import datetime
try:
    import IPython
    valids = ['inline', 'osx']
    for valid in valids:
        # print(f"Trying {valid}")
        try:
            IPython.get_ipython().run_line_magic("matplotlib", valid)
            # print(f"Using {valid}")
            break
        except:
            continue
except:
    print("")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", message="Mean of empty slice")

def prepare_dss(dss, sorter):
    for ds in dss:
        if "offset" not in ds.keys():
            ds["offset"] = None
        if "scale" not in ds.keys():
            ds["scale"] = None
        if "masks" not in ds.keys():
            ds["masks"] = None

    for ds in dss:
        to_sort = list(ds.keys())
        to_sort.append(to_sort.pop(to_sort.index(sorter)))
        for key in to_sort:
            if isinstance(ds[key], np.ndarray):
                ds[key] = ds[key].tolist()
            if not isinstance(ds[key], list):
                ds[key] = [ds[key]] * len(ds[sorter])
            ds[key] = [x for _, x in sorted(zip(ds[sorter], ds[key]))]

    for ds in dss:
        assert dss[0][sorter] == ds[sorter]

    return dss

def determine_offsets(dss, box_width = 2):
    no_datasets = len(dss)
    box_space = box_width / 4
    total_width = no_datasets * box_width + (no_datasets - 1) * box_space
    offsets = np.linspace(-total_width / 2 + box_width / 2, 
                        total_width / 2 - box_width / 2,
                        no_datasets)
    return offsets

def open_data(fh, mask, example, scale, offset, flatten = True):

    if not isinstance(example, type(None)):
        ds = PF.reproj_ds(fh, example)
    else:
        ds = gdal.Open(fh)

    if not isinstance(mask, type(None)):
        mask_ds = gdal.Open(mask)
        mask_array = mask_ds.GetRasterBand(1).ReadAsArray()
        array = ds.GetRasterBand(1).ReadAsArray()

        print("non-masked pixels (mask)", np.sum(np.isfinite(mask_array)))

        array[np.isnan(mask_array)] = np.nan
        array[mask_array != 1] = np.nan

        print("non-masked pixels (data)", np.sum(np.isfinite(array)))
    else:
        array = ds.GetRasterBand(1).ReadAsArray()

    ndv = ds.GetRasterBand(1).GetNoDataValue()

    # print("shape: {0}".format(array.shape))

    if flatten:
        flat_array = array.flatten().astype(np.float32)
    else:
        flat_array = array.astype(np.float32)

    flat_array[flat_array == ndv] = np.nan

    if not isinstance(scale, type(None)):
        flat_array *= scale

    if not isinstance(offset, type(None)):
        flat_array += offset

    return flat_array

def calc_pearson_correlation(arrays):
    # assert len(arrays) == 2

    x = arrays[0]

    rs = np.array([])

    for y in arrays[1:]:

        xmean = np.mean(x)
        ymean = np.mean(y)
        xdiff = x - xmean
        ydiff = y - ymean

        r = np.sum(xdiff * ydiff) / (np.sqrt(np.sum(xdiff**2)) * np.sqrt(np.sum(ydiff**2)))
        rs = np.append(rs, r)

    return rs

def calc_rmse(arrays):

    x = arrays[0]
    rmses = np.array([])

    for y in arrays[1:]:
        rmse = np.sqrt(np.mean((x - y)**2))
        rmses = np.append(rmses, rmse)
    
    return rmses

def calc_nash_sutcliffe(arrays):

    x = arrays[0]
    nses = np.array([])

    for y in arrays[1:]:

        # If x and y are identical return 1.0 (e.g. when all values in both x and y are zero)
        if np.all(x == y):
            nses = np.append(nses, 1.0)
        else:
            nse = 1 - (np.sum((x - y)**2) / np.sum((y - np.mean(y))**2))
            nses = np.append(nses, nse)

    return nses

def calc_rel_bias(arrays):

    x = arrays[0]
    rbs = np.array([])

    for y in arrays[1:]:

        rb = (np.sum(x) - np.sum(y)) / np.sum(x)

        rbs = np.append(rbs, rb)

    return rbs

def plot_honeycomb(dss, date, meta, output_fh = None, 
                    minmax = None, date_format = "%Y-%m", ax = None):

    if isinstance(ax, type(None)):
        plt.clf()
        fig = plt.figure(1)
        ax = fig.gca()

    arrays = list()

    for ds in dss:

        idx = ds["dates"].index(date)

        params = (ds["files"][idx], 
                 ds["masks"][idx],
                 ds["example"][idx],
                 ds["scale"][idx], 
                 ds["offset"][idx])

        array = open_data(*params)

        if not isinstance(minmax, type(None)):

            mini = np.nanpercentile(array, minmax[0])
            maxi = np.nanpercentile(array, minmax[1])

            min_mask = np.sum(array < mini)
            max_mask = np.sum(array > maxi)
            print(f"Masking {min_mask} pixels smaller than {mini}")
            array[array < mini] = np.nan
            print(f"Masking {max_mask} pixels greater than {maxi}")
            array[array > maxi] = np.nan

        arrays.append(array)

    orig_arrays = np.copy(arrays)

    mask = np.any([np.isnan(array) for array in arrays], axis = 0)
    arrays = np.array([array[~mask] for array in arrays])

    rs = calc_pearson_correlation(arrays)
    nses = calc_nash_sutcliffe(arrays)
    n = arrays[0].size

    hb = ax.hexbin(arrays[1], arrays[0], bins = "log")

    ax.set_facecolor(plt.cm.get_cmap('viridis')(0.0))

    ax.grid(color="k", linestyle=":")

    ax.set_title(meta["title"] + f"\n r: {rs[0]:.3f}, nse: {nses[0]:.3f}")
    # ax.set_title(meta["title"] )

    ax.set_xlabel(meta["labels"][1])
    ax.set_ylabel(meta["labels"][0])

    if not isinstance(output_fh, type(None)):
        plt.savefig(output_fh)

    return orig_arrays, arrays

def plot_boxplots(dss, meta, output_fh = None, box_width = 2, date_format = "%Y-%m", fig = None, ax = None, legend = True):

    offsets = determine_offsets(dss, box_width=box_width)

    if isinstance(fig, type(None)):
        plt.clf()
        plt.figure(1)
        fig = plt.gcf()

        width = (5/7) * len(dss[0]["files"])
        height = 4.2

        # fig.set_size_inches(3.6, 2.5)
        fig.set_size_inches(width, height)

    if isinstance(ax, type(None)):
        ax = fig.gca()

    for ds in dss:

        input_params = [ds["files"], ds["dates"], ds["dataset"],
                        ds["masks"], ds["scale"], ds["offset"]]
        
        for params in zip(*input_params):

            fh, date, ds_no = params[:3]

            if date in meta["exclude"] or fh in meta["exclude"]:
                continue

            flat_array = open_data(params)

            finite_array = flat_array[np.isfinite(flat_array)]

            i = date.toordinal()
            i += offsets[ds_no - 1]

            box = ax.boxplot(finite_array, positions = [i], widths = [box_width], 
                            showfliers = False, patch_artist = True, whis = (1, 99)
                            )

            clr = meta["colors"][ds_no - 1]
            box["medians"][0].set_color("black")
            box["medians"][0].set_linestyle("--")
            box["boxes"][0].set_linewidth(0.)
            box["boxes"][0].set_facecolor(clr)

            print("{0} processed.".format(date))

    xticks = [d.toordinal() for d in np.sort(np.unique(dss[0]["dates"])) if d not in meta["exclude"]]
    ax.set_xticks(xticks)
    xlabels = [d.date().strftime(date_format) for d in np.sort(np.unique(dss[0]["dates"])) if d not in meta["exclude"]]
    ax.set_xticklabels(xlabels, rotation = 20, ha = "right")

    ax.grid(zorder = 0)
    ax.set_facecolor("lightgray")
    if "xlabel" in meta.keys():
        ax.set_xlabel(meta["xlabel"])
    else:
        ax.set_xlabel("Dates")
    ax.set_ylabel(meta["ylabel"])
    ax.set_xlim([xticks[0] + offsets[0] * 2, xticks[-1] + offsets[-1] * 2])

    if legend:
        legend_elements = [Patch(facecolor = meta["colors"][ds_no - 1], label = meta["labels"][ds_no - 1]) for ds_no in np.arange(1, len(dss)+1)]
        ax.legend(handles = legend_elements, loc="best")
    ax.set_title(meta["title"])

    if not isinstance(output_fh, type(None)):
        fig.savefig(output_fh)

    return ax

def compare_two_tifs(tif1, tif2, example_tif, options1 = None, options2 = None, output = None, quant_unit = None):
    
    date = datetime.datetime(2021,1,1)

    ds1 = dict()
    ds1["files"] = [tif1]
    ds1["dates"] = [date]
    ds1["example"] = example_tif
    ds1["dataset"] = 1

    if not isinstance(options1, type(None)):
        for option, value in options1.items():
            ds1[option] = value

    ds2 = dict()
    ds2["files"] = [tif2]
    ds2["dates"] = [date]
    ds2["example"] = example_tif
    ds2["dataset"] = 2

    if not isinstance(options2, type(None)):
        for option, value in options2.items():
            ds2[option] = value

    meta = dict()
    meta["exclude"] = []
    meta["colors"] = ["r", "g"]
    meta["labels"] = [f"pyWaPOR {quant_unit[0]} [{quant_unit[1]}]", f"WaPOR {quant_unit[0]} [{quant_unit[1]}]"]
    meta["title"] = ""
    meta["ylabel"] = ""
    meta["xlabel"] = ""
    meta["sorter"] = "dates"

    dss = prepare_dss([ds1, ds2], meta["sorter"])

    ds = gdal.Open(example_tif)
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize

    nrows = 2
    ncols = 3
    fig = plt.figure(1)
    fig.clf()
    fig.set_size_inches(11.69, 6.27) # 8.27

    axs = fig.subplots(nrows, ncols, sharex=False, sharey=False)
    axs = axs.flatten()

    plt.subplots_adjust(hspace=0.5, wspace = 0.3)

    orig_arrays, arrays = plot_honeycomb(dss, date, meta, ax = axs[2])

    

    mask = np.any([np.isnan(orig_arrays[0]), np.isnan(orig_arrays[1])], axis = 0)
    orig_arrays[0][mask] = np.nan
    orig_arrays[1][mask] = np.nan

    mini = np.nanpercentile(orig_arrays,  0.01)
    maxi = np.nanpercentile(orig_arrays, 99.99)

    axs[2].set_xlim([mini, maxi])
    axs[2].set_ylim([mini, maxi])

    axs[0].hist(arrays[0], bins = int((xsize*ysize)*0.00025))
    axs[0].set_title("pyWaPOR (n = {0})".format(arrays[0].size))
    axs[0].set_facecolor("lightgray")
    axs[0].grid(zorder = 0)
    axs[0].set_xlabel(f"{quant_unit[0]} [{quant_unit[1]}]")   
    axs[0].set_ylabel("Number of pixels [-]")
    axs[0].set_xlim([mini, maxi])

    axs[1].hist(arrays[1], bins = int((xsize*ysize)*0.00025))
    axs[1].set_title("WaPOR (n = {0})".format(arrays[1].size))
    axs[1].set_facecolor("lightgray")
    axs[1].grid(zorder = 0)
    axs[1].set_xlim([mini, maxi])
    axs[1].set_xlabel(f"{quant_unit[0]} [{quant_unit[1]}]")

    shw1 = axs[3].imshow(orig_arrays[0].reshape((ysize,xsize)), vmin = mini, vmax = maxi)
    shw2 = axs[4].imshow(orig_arrays[1].reshape((ysize,xsize)), vmin = mini, vmax = maxi)
    
    error_img = orig_arrays[0].reshape((ysize,xsize)) - orig_arrays[1].reshape((ysize,xsize))
    error_lim = np.nanpercentile(np.abs(error_img), 99)

    shw3 = axs[5].imshow(error_img, vmin = -1 * error_lim, vmax = error_lim, cmap = 'coolwarm')

    xax3 = axs[3].axes.get_xaxis()
    yax3 = axs[3].axes.get_yaxis()
    xax3 = xax3.set_visible(False)
    yax3 = yax3.set_visible(False)
    axs[3].set_title(f"pyWaPOR {quant_unit[0]}")

    xax4 = axs[4].axes.get_xaxis()
    yax4 = axs[4].axes.get_yaxis()
    xax4 = xax4.set_visible(False)
    yax4 = yax4.set_visible(False)
    axs[4].set_title(f"WaPOR {quant_unit[0]}")

    xax5 = axs[5].axes.get_xaxis()
    yax5 = axs[5].axes.get_yaxis()
    xax5 = xax5.set_visible(False)
    yax5 = yax5.set_visible(False)
    axs[5].set_title(f"pyWaPOR {quant_unit[0]} - WaPOR {quant_unit[0]}")

    cax3 = axs[3].inset_axes([0.00, -0.16, 1.0, 0.04], transform=axs[3].transAxes)
    cax4 = axs[4].inset_axes([0.00, -0.16, 1.0, 0.04], transform=axs[4].transAxes)
    cax5 = axs[5].inset_axes([0.00, -0.16, 1.0, 0.04], transform=axs[5].transAxes)

    fig.colorbar(shw1, cax = cax3, ax = axs[3], extend = "both", orientation = "horizontal")
    fig.colorbar(shw2, cax = cax4, ax = axs[4], extend = "both", orientation = "horizontal")
    fig.colorbar(shw3, cax = cax5, ax = axs[5], extend = "both", orientation = "horizontal")

    fig.suptitle(f'{os.path.split(tif1)[-1]} vs. {os.path.split(tif2)[-1]}')

    if not isinstance(output, type(None)):
        fig.savefig(output)

    return np.nanmean(orig_arrays[0]), np.nanmean(orig_arrays[1]), [orig_arrays[0].reshape((ysize,xsize)), orig_arrays[1].reshape((ysize,xsize))]

def plot_ts(ax, times, values, sources, styling):

    for source in np.unique(sources):

        if source == -1:
            continue

        xs = times[sources == source]
        ys = values[sources == source]

        if np.sum(np.isnan(ys)) == ys.size:
            continue

        c_styling = styling[source]

        ax.scatter(xs, ys, marker = c_styling[0], c = c_styling[1], label = c_styling[3], zorder = 10, alpha = float(c_styling[2]))
        ax.set_facecolor("lightgray")
        ax.grid(color="k", linestyle=":")
        ax.legend(loc = "best")
        ax.set_title("RAW Values")

    # ax.set_ylabel("NDVI [-]")

    return ax

def plot_composite_ts(ax, starts, ends, values, label = "Composite"):

    ax.bar(starts, values, (ends - starts), align = "edge", color = "lightblue", edgecolor = "darkblue", label = label)
    ax.set_facecolor("lightgray")
    ax.grid(color="k", linestyle=":")
    # ax.set_ylabel("NDVI [-]")
    ax.set_title("Composite Values")
    ax.legend(bbox_to_anchor=(1,1), loc="upper left")

def check_bb(ds, diagnostics_lon, diagnostics_lat):
    lon_lim = [float(ds.lon.min().values), 
                float(ds.lon.max().values)]
    lat_lim = [float(ds.lat.min().values), 
                float(ds.lat.max().values)]
    lon_test = np.all([diagnostics_lon >= lon_lim[0],
                       diagnostics_lon <= lon_lim[1]])
    lat_test = np.all([diagnostics_lat >= lat_lim[0],
                       diagnostics_lat <= lat_lim[1]])
    if not lon_test:
        print(f"WARNING: {diagnostics_lon} not in boundingbox ({lon_lim}).")
    if not lat_test:
        print(f"WARNING: {diagnostics_lat} not in boundingbox ({lat_lim}).")

def plot_composite(ds, diagnostics, out_folder = None):

    ds = ds.reindex({"y": ds.y.sortby("y"), 
                     "x": ds.x.sortby("x")})

    var = "_".join([x for x in ds.data_vars if "_values" in x][0].split("_")[0:-1])

    styling = dict()
    markers = ["*", "o", "v", "s", "*", "p", "h"]
    colors =  ["r", "g", "b", "y", "purple", "darkblue", "gray", "orange"]
    for i, source_name in ds[f"{var}_source"].attrs.items():
        i = int(i)
        styling[i] = (markers[i], colors[i], 1.0, source_name)
    styling[255] = (".", "k", 0.7, "Interp.")

    for point, (diagnostics_lat, diagnostics_lon) in diagnostics.items():

        ts = ds.sel(x = diagnostics_lon, 
                    y = diagnostics_lat, method="nearest")

        times = ts.time.values
        ndvis = ts[f"{var}_values"].values
        sources = ts[f"{var}_source"].values
        ndvi_composites = ts[var].values

        epoch_starts = ts.time_bins.values
        epoch_ends = np.append(epoch_starts[1:], np.datetime64(ds.bin_end))

        fig = plt.figure(1)
        fig.clf()
        fig.set_size_inches(9, 5)
        ax = fig.gca()

        plot_ts(ax, times, ndvis, sources, styling)
        plot_composite_ts(ax, epoch_starts, epoch_ends, ndvi_composites, f"Composite ({ds.attrs['comp_type']})")

        ax.set_ylabel(f"{var}")
        ax.set_title(f"{var} at {point}")

        fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
        fig.suptitle(f"{diagnostics_lon} °E, {diagnostics_lat} °N")
        fig.subplots_adjust(right=0.75)

        if not isinstance(out_folder, type(None)):
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            fn = f"{point}_{var}.png"
            fig.savefig(os.path.join(out_folder, fn))

def plot_tif(tif_file, quantity = None, unit = None):
    ds = gdal.Open(tif_file)
    array = ds.GetRasterBand(1).ReadAsArray()
    ndv = ds.GetRasterBand(1).GetNoDataValue()

    array[array == ndv] = np.nan

    mini = np.nanpercentile(array, 5)
    maxi = np.nanpercentile(array, 95)

    fn = os.path.split(tif_file)[-1]

    plt.imshow(array, vmin = mini, vmax = maxi)
    plt.colorbar(label = f"{quantity} {unit}", extend = "both")
    plt.title(fn)
    plt.gca().set_facecolor("lightgray")

def prettyprint(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         prettyprint(value, indent+1)
      else:
         print('\t' * (indent) + " " + str(value))

def plot_img(ax, array, title = "", var_name = "", cmap = "viridis", cb_limits = None):
    if cmap == "coolwarm":
        max_error = np.nanpercentile(np.abs(array), 99)
        vmin = -max_error
        vmax = max_error
    else:
        vmin = None
        vmax = None
    if not isinstance(cb_limits, type(None)):
        vmin = cb_limits[0]
        vmax = cb_limits[1]
    img0 = ax.imshow(array, cmap = cmap, vmin = vmin, vmax = vmax)
    cax0 = ax.inset_axes([0.00, -0.20, 1.0, 0.04], transform=ax.transAxes)
    ax.get_figure().colorbar(img0, ax = ax, cax = cax0, extend = "both", 
                orientation = "horizontal", label = var_name)
    ax.set_title(title)

def plot_hexbin(ax, arrays, xlabel = "", ylabel = "", title = ""):
    minmax = [np.min(arrays), np.max(arrays)]
    ax.plot(minmax, minmax, ":k", label = "1:1")
    ax.legend()
    hb = ax.hexbin(*arrays, bins = "log")
    ax.set_facecolor(plt.cm.get_cmap('viridis')(0.0))
    ax.get_figure().colorbar(hb, label = "Number of pixels [-]")
    ax.set_xlim(minmax)
    ax.set_ylim(minmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

# import geoviews as gv
# import holoviews as hv
# hv.extension('bokeh')

# def make_geoview(ds):
#     da = ds.to_array("variable")

#     gv_ds = gv.Dataset(da, vdims = ["data"])

#     out = gv_ds.to(gv.Image, ["x", "y"]).opts(colorbar = True, width = 500, height = 500)

#     return out


# styling[999] = ("P", "orange", 1.0, "-")
# styling[0] = (".", "k", 0.7, "Interp.")
# sources = {v[3]: k for k, v in styling.items()}
