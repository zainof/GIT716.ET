import xarray as xr
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from pywapor.enhancers.smooth.whittaker import whittaker, cross_val_lmbda, second_order_diff_matrix, cve1, cve2, whittaker_smoothing
from pywapor.enhancers.smooth.core import wt1, wt2
import pandas as pd
import os
import itertools
import pywapor

def filter_ts(x, y, u = None, tol = 0.002, chunks = "auto", sample_interval = 1, xdim = "time"):

    if not isinstance(u, type(None)):
        if not isinstance(u, xr.DataArray):
            u = xr.DataArray(u, dims = x.dims, coords = x.coords)

    if sample_interval > 1:
        x = x.isel({xdim: slice(None, None, sample_interval)})
        y = y.isel({xdim: slice(None, None, sample_interval)})

    diff_forward = x.diff(xdim).assign_coords({xdim: x.time[:-1]})
    diff_backward = x.diff(xdim).assign_coords({xdim: x.time[1:]})
    d2x = (diff_forward + diff_backward).reindex_like(x.time, fill_value = tol * 2)
    
    nremoving = (d2x < tol).sum().values
    if nremoving > 0:
        log.info(f"--> Removing {nremoving} {xdim} slices.")

    if not isinstance(u, type(None)):
        if sample_interval > 1:
            u = u.isel({xdim: slice(None, None, sample_interval)})
        u = u.where(d2x >= tol, drop = True)
        u = u.values

    y = y.where(d2x >= tol, drop = True).chunk(chunks)
    x = x.where(d2x >= tol, drop = True)

    return x, y, u

def open_ts(fps):

    dss = list()
    for i, fp in enumerate(fps):

        name = os.path.split(fp)[-1].replace(".nc", "")
        ds = xr.open_dataset(fp, decode_coords = "all")

        if i > 0:
            if ds.x.size != example_ds.x.size or ds.y.size != example_ds.y.size:
                ds = ds.rio.reproject_match(example_ds)

        ds["sensor"] = xr.ones_like(ds["time"], dtype = int).where(False, name)
        dss.append(ds)

        if i ==0:
            example_ds = ds.copy()
    
    ds = xr.concat(dss, dim = "time", combine_attrs = "drop_conflicts").sortby("time").transpose("y", "x", "time")
    ds["sensor"] = ds["sensor"].astype("<U7")
    ds.attrs = {}

    attribute = {str(i): sensor_name for i, sensor_name in enumerate(np.unique(ds.sensor))}
    values = np.array(list(attribute.keys()), dtype = int)
    coords = np.array(list(attribute.values()), dtype = str)
    transformer = xr.DataArray(values, dims=["sensor"], coords = {"sensor": coords})
    ds["sensor"] = transformer.sel(sensor = ds.sensor).drop("sensor").assign_attrs(attribute)

    return ds

def cross_val_lmbda(y, x, lmbdas = np.logspace(0, 3, 4), 
                    u = None, chunks = "auto", tol = 0.02,
                    sample_interval = 1):

    x, y, lmbdas, dim_name = assert_dtypes(x, y, lmbdas)

    # Normalize x-coordinates
    x = (x - x.min()) / (x.max() - x.min()) * x.size

    # Remove points that are too close together
    if isinstance(x, xr.DataArray):
        x, y, u = filter_ts(x, y, u = u, tol = tol, chunks = chunks, 
                            sample_interval = sample_interval, xdim = dim_name)

    # Create x-aware delta matrix.
    D = second_order_diff_matrix(x)
    A = np.dot(D.T, D)

    # Choose which function to use depending on shapes of y and lmbdas.
    # cve = choose_func(y, lmbdas, "cve")

    if isinstance(u, type(None)):
        u = np.ones_like(x)

    if isinstance(x, xr.DataArray):

        # # Determine dimension names.
        # if cve.__name__ == "cve1":
        #     icd = [[], [dim_name], [], []]
        #     ocd = [[]]
        # elif cve.__name__ == "cve2":
        #     icd = [["lmbda"], [dim_name], [], []]
        #     ocd = [["lmbda"]]

        # Make sure lmbdas is chunked similar to y.
        if not isinstance(y.chunk, type(None)):
            lmbdas = lmbdas.chunk({k: v for k,v in y.unify_chunks().chunksizes.items() if k in lmbdas.dims})

        # NOTE this is done to supress a warning ("invalid value encountered in cve2")
        # which is given when a column (along `dim_name`) doenst contain any finite value.
        # Its currently impossible to supress this warning, so filling those columns with
        # fake data (-99) here and then masking those pixels again afterwards.
        # See https://github.com/dask/dask/issues/3245 for more info.
        if cve.__name__ == "cve2":
            mask = y.count(dim_name) > 0
            y = y.where(mask, -99)

        # Calculate the cross validation standard error for each lambda using
        # the hat matrix.
        cves = xr.apply_ufunc(
            cve, lmbdas, y, A, u,
            input_core_dims=icd,
            output_core_dims=ocd,
            dask = "allowed")

        # NOTE see above.
        if cve.__name__ == "cve2":
            cves = cves.where(mask, np.nan)

        # Select the lambda for which the error is smallest.
        if "lmbda" in cves.dims:
            lmbda_sel = cves.idxmin(dim = "lmbda")
            lmbda_sel = lmbda_sel.fillna(lmbdas.min().values)
        else:
            cves.assign_coords({"lmbda": lmbdas})
            lmbda_sel = lmbdas
    else:
        cves = cve(lmbdas, y, A, u)
        if np.isscalar(lmbdas):
            lmbda_sel = lmbdas
        elif lmbdas.ndim == 1:
            idx = np.argmin(cves, axis = -1)
            lmbda_sel = lmbdas[idx]
        else:
            lmbda_sel = lmbdas
        
    return lmbda_sel, cves

def cross_val_lmbda_ref(y, x, lmbdas = np.logspace(-2, 3, 100)):
    # Determine vector length.
    m = len(x)
    # Create arrays of m timeseries in which each time one value has been removed.
    Y_ = np.where(np.eye(m), np.nan, np.repeat(y[np.newaxis, :], m, axis = 0))
    # Smooth the timeeseries and make an estimate for the removed value for each lambda.
    z = whittaker(Y_, x, lmbdas = lmbdas)
    # Retrieve estimated timeseries for each lambda.
    y_hat = np.diagonal(z, axis1 = 0, axis2 = 2)
    # Calculate the cross validation standard error for each lambda.
    cves = np.sqrt(np.nanmean(((y - y_hat)**2), axis = 1))
    # Select the lambda for which the error is smallest.
    lmbda_sel = lmbdas[np.argmin(cves)]
    return lmbda_sel, cves

def test_cross_val(y, x):
    lmbdas = np.logspace(-2, 2, 20)
    lmbda_sel1, cves1 = cross_val_lmbda_ref(y, x, lmbdas = lmbdas)
    lmbda_sel2, cves2 = cross_val_lmbda(y, x, lmbdas = lmbdas)
    # assert np.isclose(lmbda_sel1, lmbda_sel2)
    plt.plot(lmbdas, cves1, label = "1")
    plt.plot(lmbdas, cves2, label = "2")
    plt.legend()

def test_shapes_z(y, x, xn = 2, yn = 5, n = 4, m = 3, a = 0.5, max_iter = 10):
    t = len(y)

    Y1 = y[:]
    Y2 = np.repeat(y, xn*yn).reshape((xn,yn,t), order = "F")
    Y3 = np.repeat(y, n).reshape((n,t), order = "F")

    LMB1 = 100.
    LMB2 = np.logspace(1,3,m)
    LMB3 = np.logspace(1,3,xn*yn).reshape((xn, yn))

    # Normalize x-coordinates
    x = (x - x.min()) / (x.max() - x.min()) * x.size

    # Create x-aware delta matrix.
    D = second_order_diff_matrix(x)
    A = np.dot(D.T, D)

    u = np.ones_like(x)

    z11 = wt1(Y1, A, LMB1, u, a, -np.inf, np.inf, max_iter)
    assert z11.shape == (t,)

    z21 = wt1(Y2, A, LMB1, u, a, -np.inf, np.inf, max_iter)
    assert z21.shape == (xn,yn,t)

    z31 = wt1(Y3, A, LMB1, u, a, -np.inf, np.inf, max_iter)
    assert z31.shape == (n,t)

    z12 = wt1(Y1, A, LMB2, u, a, -np.inf, np.inf, max_iter) #
    assert z12.shape == (m,t)

    z22 = wt2(Y2, A, LMB2, u, a, -np.inf, np.inf, max_iter) #
    assert z22.shape == (xn,yn,m,t)

    z32 = wt2(Y3, A, LMB2, u, a, -np.inf, np.inf, max_iter) #
    assert z32.shape == (n,m,t)

    z13 = wt1(Y1, A, LMB3, u, a, -np.inf, np.inf, max_iter) #
    assert z13.shape == (xn,yn,t)

    z23 = wt1(Y2, A, LMB3, u, a, -np.inf, np.inf, max_iter) #!
    assert z23.shape == (xn,yn,t)

def test_shapes_cve(y, x, xn = 2, yn = 5, n = 4, m = 3):
    t = len(y)

    Y1 = y[:]
    Y2 = np.repeat(y, xn*yn).reshape((xn,yn,t), order = "F")
    Y3 = np.repeat(y, n).reshape((n,t), order = "F")

    LMB1 = 100.
    LMB2 = np.logspace(1,3,m)
    LMB3 = np.logspace(1,3,xn*yn).reshape((xn, yn))
    
    # Normalize x-coordinates
    x = (x - x.min()) / (x.max() - x.min()) * x.size

    # Create x-aware delta matrix.
    D = second_order_diff_matrix(x)
    A = np.dot(D.T, D)

    u = np.ones_like(x)

    # cve11 = cve1(LMB1, Y1, A, u)
    # assert cve11.shape == ()

    # cve21 = cve1(LMB1, Y2, A, u)
    # assert cve21.shape == (xn,yn)

    # cve31 = cve1(LMB1, Y3, A, u)
    # assert cve31.shape == (n,)

    cve12 = cve1(LMB2, Y1, A, u)
    assert cve12.shape == (m,)

    cve22 = cve2(LMB2, Y2, A, u)
    assert cve22.shape == (xn,yn,m)

    cve32 = cve2(LMB2, Y3, A, u)
    assert cve32.shape == (n,m)

    cve13 = cve1(LMB3, Y1, A, u)
    assert cve13.shape == (xn,yn)

    cve23 = cve1(LMB3, Y2, A, u)
    assert cve23.shape == (xn,yn)

def test_whittaker_main(ds, base = 10, xn = 2, yn = 5, n = 4, m = 3):

    t = len(ds.time)
    X = ds.time

    Y1 = ds.isel(y = base, x = base)["ndvi"]
    Y2 = ds.isel(y = slice(base, base + yn), x = slice(base, base + xn))["ndvi"].transpose("y","x","time")
    Y3 = ds.isel(y = slice(base, base + n), x = base).rename({"y": "n"})["ndvi"]

    u = np.ones((t)) * 0.5

    LMB1 = xr.DataArray(100.)
    LMB2 = xr.DataArray(np.logspace(1,3,m), dims = ["lmbda"], coords = {"lmbda": np.logspace(1,3,m)})
    LMB3 = xr.DataArray(np.logspace(1,3,xn*yn).reshape((xn, yn)), dims = ["x", "y"], coords = {"x": Y2.x, "y": Y2.y}).transpose("y","x")
    
    out1 = whittaker(Y1, X, lmbdas = LMB1, u = u)
    assert isinstance(out1, xr.DataArray)
    assert np.all([v == {"time": t}[k] for k, v in out1.sizes.items()])

    out2 = whittaker(Y2, X, lmbdas = LMB1, u = u)
    assert isinstance(out2, xr.DataArray)
    assert np.all([v == {"x": xn, "y": yn, "time": t}[k] for k, v in out2.sizes.items()])

    out3 = whittaker(Y3, X, lmbdas = LMB1, u = u)
    assert isinstance(out3, xr.DataArray)
    assert np.all([v == {"n": n, "time": t}[k] for k, v in out3.sizes.items()])

    out4 = whittaker(Y1, X, lmbdas = LMB2, u = u)
    assert isinstance(out4, xr.DataArray)
    assert np.all([v == {"lmbda": m, "time": t}[k] for k, v in out4.sizes.items()])

    out5 = whittaker(Y2, X, lmbdas = LMB2, u = u)
    assert isinstance(out5, xr.DataArray)
    assert np.all([v == {"lmbda": m, "time": t, "x": xn, "y": yn}[k] for k, v in out5.sizes.items()])

    out6 = whittaker(Y3, X, lmbdas = LMB2, u = u)
    assert isinstance(out6, xr.DataArray)
    assert np.all([v == {"lmbda": m, "time": t, "n": n}[k] for k, v in out6.sizes.items()])

    out7 = whittaker(Y1, X, lmbdas = LMB3, u = u)
    assert isinstance(out7, xr.DataArray)
    assert np.all([v == {"x": xn, "time": t, "y": yn}[k] for k, v in out7.sizes.items()])

    out8 = whittaker(Y2, X, lmbdas = LMB3, u = u)
    assert isinstance(out8, xr.DataArray)
    assert np.all([v == {"x": xn, "time": t, "y": yn}[k] for k, v in out8.sizes.items()])

    ######

    out1 = whittaker(Y1.values, X.values, lmbdas = LMB1.values, u = u)
    assert out1.shape == (t,)

    out2 = whittaker(Y2.values, X.values, lmbdas = LMB1.values, u = u)
    assert out2.shape == (yn,xn,t)

    out3 = whittaker(Y3.values, X.values, lmbdas = LMB1.values, u = u)
    assert out3.shape == (n,t)

    out4 = whittaker(Y1.values, X.values, lmbdas = LMB2.values, u = u)
    assert out4.shape == (m,t)

    out5 = whittaker(Y2.values, X.values, lmbdas = LMB2.values, u = u)
    assert out5.shape == (yn,xn,m,t)

    out6 = whittaker(Y3.values, X.values, lmbdas = LMB2.values, u = u)
    assert out6.shape == (n,m, t)

    out7 = whittaker(Y1.values, X.values, lmbdas = LMB3.values, u = u)
    assert out7.shape == (yn,xn,t)

    out8 = whittaker(Y2.values, X.values, lmbdas = LMB3.transpose("y","x").values, u = u)
    assert out8.shape == (yn,xn,t)

def test_cve_main(ds, base = 10, xn = 2, yn = 5, n = 4, m = 3):

    X = ds.time

    Y1 = ds.isel(y = base, x = base)["ndvi"]
    Y2 = ds.isel(y = slice(base, base + yn), x = slice(base, base + xn))["ndvi"].transpose("y","x","time")
    Y3 = ds.isel(y = slice(base, base + n), x = base).rename({"y": "n"})["ndvi"]

    u = np.ones((X.size)) * 0.5

    LMB1 = xr.DataArray(100.)
    LMB2 = xr.DataArray(np.logspace(1,3,m), dims = ["lmbda"], coords = {"lmbda": np.logspace(1,3,m)})
    LMB3 = xr.DataArray(np.logspace(1,3,xn*yn).reshape((xn, yn)), dims = ["x", "y"], coords = {"x": Y2.x, "y": Y2.y}).transpose("y","x")
    
    out1, cves1 = cross_val_lmbda(Y1, X, lmbdas = LMB1, u = u)
    assert out1 == LMB1
    assert cves1.ndim == 0

    out2, cves2 = cross_val_lmbda(Y2, X, lmbdas = LMB1, u = u)
    assert out2 == LMB1
    assert np.all([v == {"x": xn, "y": yn,}[k] for k, v in cves2.sizes.items()])

    out3, cves3 = cross_val_lmbda(Y3, X, lmbdas = LMB1, u = u)
    assert out3 == LMB1
    assert np.all([v == {"n": n}[k] for k, v in cves3.sizes.items()])

    out4, cves4 = cross_val_lmbda(Y1, X, lmbdas = LMB2, u = u)
    assert np.all([v == {"lmbda": m}[k] for k, v in cves4.sizes.items()])
    assert out4.ndim == 0

    out5, cves5 = cross_val_lmbda(Y2, X, lmbdas = LMB2, u = u)
    assert np.all([v == {"lmbda": m, "x": xn, "y": yn}[k] for k, v in cves5.sizes.items()])
    assert np.all([v == {"x": xn, "y": yn}[k] for k, v in out5.sizes.items()])

    out6, cves6 = cross_val_lmbda(Y3, X, lmbdas = LMB2, u = u)
    assert np.all([v == {"lmbda": m, "n": n}[k] for k, v in cves6.sizes.items()])
    assert np.all([v == {"n": n}[k] for k, v in out6.sizes.items()])

    out7, cves7 = cross_val_lmbda(Y1, X, lmbdas = LMB3, u = u)
    assert out7.equals(LMB3)
    assert np.all([v == {"x": xn, "y": yn}[k] for k, v in cves7.sizes.items()])

    out8, cves8 = cross_val_lmbda(Y2, X, lmbdas = LMB3, u = u)
    assert out8.equals(LMB3)
    assert np.all([v == {"x": xn, "y": yn}[k] for k, v in cves8.sizes.items()])

    ######

    out1, cves1 = cross_val_lmbda(Y1.values, X.values, lmbdas = LMB1.values, u = u)
    assert out1 == LMB1.values
    assert np.isscalar(cves1)

    out2, cves2 = cross_val_lmbda(Y2.values, X.values, lmbdas = LMB1.values, u = u)
    assert out3 == LMB1.values
    assert cves2.shape == (yn,xn)

    out3, cves3 = cross_val_lmbda(Y3.values, X.values, lmbdas = LMB1.values, u = u)
    assert out3 == LMB1.values
    assert cves3.shape == (n,)

    out4, cves4 = cross_val_lmbda(Y1.values, X.values, lmbdas = LMB2.values, u = u)
    assert np.isscalar(out4)
    assert cves4.shape == (m,) 

    out5, cves5 = cross_val_lmbda(Y2.values, X.values, lmbdas = LMB2.values, u = u)
    assert out5.shape == (yn, xn)
    assert cves5.shape == (yn,xn,m) 

    out6, cves6 = cross_val_lmbda(Y3.values, X.values, lmbdas = LMB2.values, u = u)
    assert out6.shape == (n,)
    assert cves6.shape == (n, m)  

    out7, cves7 = cross_val_lmbda(Y1.values, X.values, lmbdas = LMB3.values, u = u)
    assert np.all(out7 == LMB3.values)
    assert cves7.shape == (yn, xn)  

    out8, cves8 = cross_val_lmbda(Y2.values, X.values, lmbdas = LMB3.values, u = u)
    assert np.all(out8 == LMB3.values)
    assert cves8.shape == (yn, xn)  

def test_whittaker_smoothing(ds, var, m = 3):

    xn = ds.x.size
    yn = ds.y.size

    LMB0 = 100.
    LMB1 = xr.DataArray(100.)
    LMB2 = xr.DataArray(np.logspace(1,3,m), dims = ["lmbda"], coords = {"lmbda": np.logspace(1,3,m)})
    LMB3 = xr.DataArray(np.logspace(1,3,xn*yn).reshape((xn, yn)), dims = ["x", "y"], coords = {"x": ds.x, "y": ds.y}).transpose("y","x")
    LMB4 = LMB1.values
    LMB5 = LMB2.values

    _LMB = [
            LMB0, 
            LMB1, LMB2, 
            LMB3, LMB4, LMB5
            ]

    out_fh = None
    xdim = "time"
    new_x1 = None
    new_x2 = pd.date_range(np.datetime_as_string(ds.time[0], "D") + " 12:00", 
                           np.datetime_as_string(ds.time[-1], "D") + " 12:00", freq="D")
    _new_x = [new_x1, new_x2]
    _export_all = [True, False]
    _DS = [
        ds.load(), 
        ds.chunk({"x": 5, "y": 5, "time": -1})
        ]

    _a = [0.5, 0.1]

    _u = [None, {"LC08_SR": 1.0, "LC09_SR": 1.0, "LE07_SR": 0.7}]

    tests = list(itertools.product(*[_DS, _LMB, _new_x, _export_all, _a, _u]))

    files = list()

    for i, (DS, LMB, new_x, export_all, a, u) in enumerate(tests):

        print(f"TEST {i:>03} / {len(tests):>03}")
        print("ds", DS)
        print("lmbda", LMB)
        print("new_x", new_x)
        print(export_all)
        print(a)
        print(u)
        print("------\n")

        out = whittaker_smoothing(DS, var,
                                lmbdas = LMB,
                                weights = u,
                                a = a,
                                max_iter = 10,
                                out_fh = out_fh,
                                xdim = xdim,
                                new_x = new_x,
                                export_all = export_all)
        
        out = out.compute(scheduler = "synchronous")

        if export_all:
            assert np.all(np.sort(list(out.data_vars)) == np.sort(["ndvi", "sensor", "lmbda_sel", "cves", "ndvi_smoothed"]))
            assert "lmbda" in out.coords
            assert out.time.size == ds.time.size + getattr(new_x, "size", 0)
        else:
            assert (out.time.size == new_x2.size) or (out.time.size == ds.time.size)
            assert list(out.data_vars) == ["ndvi_smoothed"]

        files.append(out_fh)
        out = out.close()
        # os.remove(fp)

    for fp in files:
        if not isinstance(fp, type(None)):
            if os.path.isfile(fp):
                os.remove(fp)

if __name__ == "__main__":

    ts_fp = os.path.join(pywapor.__path__[0], "enhancers", "smooth", "input_test_series.nc")

    ds = xr.open_dataset(ts_fp, decode_coords="all")
    var = "ndvi"
    y = ds[var]
    x = ds["time"]

    m = 3

    test_shapes_z(y.isel(y=10, x=10).values, x.values)
    test_shapes_cve(y.isel(y=10, x=10).values, x.values)
    # test_whittaker_main(ds)
    # test_cve_main(ds)
    # test_whittaker_smoothing(ds, var)