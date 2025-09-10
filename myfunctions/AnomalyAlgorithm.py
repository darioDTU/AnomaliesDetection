import numpy as np
import xarray as xr
from scipy import stats
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
from bokeh.models import FixedTicker
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from myfunctions.coordinates_values import *

def ClimatologyCalculation(baseline, dataset):

    # compute the mean across the 'time' dimension
    climatology = baseline.groupby("time.month").mean("time", skipna=True)
    climatology = climatology.squeeze()
    climatology = climatology['thetao']
    da4d = dataset['thetao']

    return da4d, climatology

def AnomalyDetection(da4d, climatology):
    # clim0 = climatology.isel(year=0, drop=True)½
    # climatology = climatology.assign_coords(month=climatology['month'].astype(int))
    # climatology = climatology.to_array(dim='var')
    # data = data.to_array(dim='var')
    data_grouped = da4d.groupby("time.month").mean()
    deviation = data_grouped - climatology
    # deviation = clim0 #TODO put deviation here if there are multiple years
    vals = deviation.load()  
    vals_array = abs(vals.data)
    valid = vals_array[np.isfinite(vals_array)]

    if valid.size == 0:
        raise ValueError("No finite deviations found! Check your data/climatology overlap.")
    threshold = float(np.nanpercentile(valid, percentile))
    mask = deviation >= threshold
    anomalies = deviation.where(mask)

    threshold_array = climatology + threshold
    return deviation, threshold_array

def POT(X, q=0.01, t=None, t_pct=percentile/100):
    """
    Peaks-Over-Threshold value estimator

    In:
        -> X : Calibration data
        -> q : tail probability for false-positive
        -> t : initial thresold
        -> t_pct : quantile level to pick when t is None
    Out :
        -> zq : estimated threshold
        -> t : threshold for GPD fitting
    """
    X = np.asarray(X)
    # drop NaNs
    X = X[np.isfinite(X)]
    n = X.size
    if n == 0:
        raise ValueError("Empty input array after dropping NaNs!")

    # initial threshold
    if t is None:
        t = float(np.percentile(X, t_pct))

    # excesses
    excesses = X[X > t] - t
    Nt = excesses.size
    if Nt == 0:
        raise ValueError(
            f"No data points exceed t={t:.3g}! "
            "Try lowering t_pct or set t explicitly."
        )

    # fit GPD
    γ̂, loc, σ̂ = stats.genpareto.fit(excesses, floc=0)

    # compute zq
    if abs(γ̂) < 1e-6:
        zq = t + σ̂* np.log(Nt / (q * n))
    else:
        zq = t + (σ̂/ γ̂) * ((q * n / Nt) ** (-γ̂) - 1)

    return zq, t

def DSPOT(X, d, n=5000, q=0.01):
    """
    Streaming Peaks-Over-Threshold with drift algorithm

    In :
        -> X : 6D array (climatology)
        -> n : calibration window size (must be =< len(x))
        -> d : local Depth for drift estimation
        -> q : tail probability controlling false-positive rate
    Out :
        -> anomalies : boolean mask where True indicates an anomaly
        -> zq : final threshold
        -> initial threshold
    """
    # flatten input to 1D
    X = np.asarray(X).ravel()
    if X.size < n:
        raise ValueError(f"Need at least {n} initial samples, got {X.size}")

    # initial calibration using POT on first n samples
    cal = X[:n]
    zq, t = POT(cal, q, t_pct=percentile/100)

    # Record initial peaks above t
    peaks = list(cal[cal > t] - t)

    # anomaly mask
    anomalies = np.zeros(X.shape, dtype=bool)

    # drift window with last d calibration samples
    window = deque(cal[-min(d, n):], maxlen=d)

    for i in range(n, X.size):
        # remove drift
        r = X[i] - np.mean(window)

        # anomaly if above current zq
        if r > zq:
            anomalies[i] = True
        elif r > t:
            peak = r - t
            peaks.append(peak)
            arr_peaks = np.array(peaks)
            gamma, _, sigma = stats.genpareto.fit(arr_peaks, floc=0)
            Nt = arr_peaks.size
            # recompute zq
            if abs(gamma) < 1e-6:
                zq = t + sigma * np.log(Nt / (q * n))
            else:
                zq = t + (sigma / gamma) * ((q * n / Nt) ** (-gamma) - 1)
        # slide drift window
        window.append(X[i])

    return anomalies, zq, t

def ProcessAnomalies(da6d, climatology, res):
    # algorithms = ['Classic', 'POT', 'DSPOT']
    # print("Select anomaly processing:")
    # for idx, name in enumerate(algorithms):
    #     print(f"[{idx}] {name}")

    # res = None
    # while res not in {0,1,2}:
    #     try:
    #         res = int(input('Enter 0, 1, or 2: '))
    #     except ValueError:
    #         print("Invalid input, please enter a number.")

    threshold = None
    if res == 0:
        _, threshold = AnomalyDetection(da6d, climatology)
    elif res == 1:
        climatology = climatology.to_array(dim='var')
        threshold, _ = POT(climatology)
    else:
        _, threshold, _ = DSPOT(X=climatology, d=max_depth)

    return threshold

def DetectAnomalies(da6d, threshold):
    """
    Flags all entries in the deviation array that exceed threshold zq
    
    In :
        -> Deviation : 6D array
        -> zq : estiamted thresold in the algorithm
    Out :
        -> mask : array same shape as deviation with True when deviation > zq
        -> coords : array like (n_anomalies, ndim) with coordinates of all anomalies
    """
    # arr = np.asarray(da6d)
    mask = da6d > threshold
    coords = np.argwhere(mask)

    print(f"Found {coords.shape[0]} anomalies out of {da6d.size} total points.")

    return mask, coords   

def ShowPixelAnomalies(mask, var_index=None):
    """
    Plot the total anomaly count over lon/lat.

    In:
      mask      : boolean array (year, month, lon, lat, depth, var)
      var_index : int or None, if set selects only that variable
    """

    print(f'{len(variable)} number of variable :')
    for i in range(len(variable)):
        print(f'[{i}] for {variable[i]}')
    var_index = int(input(('Please select the one you want to see the anomalies : ')))

    # build coordinate edges from variables
    lon_edges = np.arange(min_longitude,
                          max_longitude + area_resolution,
                          area_resolution)
    lat_edges = np.arange(min_latitude,
                          max_latitude + area_resolution,
                          area_resolution)
    depth_edges = np.arange(min_depth,
                            max_depth + depth_resolution,
                            depth_resolution)
    depth_centers = depth_edges[:-1] + depth_resolution/2
    depth_range = (0,depth_resolution)

    # dims
    n_year, n_month, n_lon, n_lat, n_depth, n_var = mask.shape

    # select depth indices
    if depth_range is None:
        sel_depth = slice(None)
    else:
        dmin, dmax = depth_range
        sel_depth = np.where((depth_centers >= dmin) & (depth_centers <= dmax))[0]
        if sel_depth.size == 0:
            raise ValueError(f"No depth bins in range {dmin}-{dmax}m.")

    total_anomaly_count = np.zeros((n_lon, n_lat), dtype=int)

    for y in range(n_year):
        for m in range(n_month):
            slice4d = mask[y, m]  # (lon, lat, depth, var)
            if var_index is None:
                # sum over selected depths & all variables
                counts = slice4d[:, :, sel_depth, :].sum(axis=(2,3))
            else:
                if not (0 <= var_index < n_var):
                    raise IndexError(f"var_index {var_index} out of range [0,{n_var-1}]")
                counts = slice4d[:, :, sel_depth, var_index].sum(axis=2)
            total_anomaly_count += counts

    # plotting on lon/lat
    extent = [lon_edges[0], lon_edges[-1], lat_edges[0], lat_edges[-1]] # you can also remove this variable 
                                                                        # as well as in the imshow in you want to show index
    plt.figure(figsize=(8,6))
    im = plt.imshow(
        total_anomaly_count.T,
        origin='lower',
        extent=extent,
        aspect='auto',
        cmap='YlGn',
        interpolation='nearest'
    )
    plt.colorbar(im, label='Total anomalies')
    plt.xlabel('Longitude (°E)')
    plt.ylabel('Latitude (°N)')
    plt.xlim(extent[0], extent[1])
    plt.ylim(extent[2], extent[3])
    plt.xlim()
    title = 'All-time anomaly count'
    if var_index is not None:
        title += f' (variable {variable[var_index]})'
    #if depth_range is not None:
    #    title += f' (depth {variable[var_index]} m)'
    plt.title(title)
    plt.show()

    return var_index

def ShowGraphAnomaly(da6d, mask, threshold, var_index):

    output_notebook()
    
    # creation of list containing all longitude and latitude, same size as each lon/lat in da6d
    # this is a more "user-friendly" way to enter coordinate since in da6d the coordinates are not in degres
    #lon_vals = np.arange(min_longitude,max_longitude + area_resolution/2,area_resolution)
    #lat_vals = np.arange(min_latitude,max_latitude + area_resolution/2,area_resolution)

    user_lon = float(input("Enter longitude (Array Index): "))
    user_lat = float(input("Enter latitude  (Array Index): "))

    # take the closest value from the list
    #user_lon = np.abs(lon_vals - user_lon).argmin()
    #user_lat = np.abs(lat_vals - user_lat).argmin()

    depth_k = 0
    var_idx = var_index

    # creation of list containing all longitude and latitude, same size as each lon/lat in da6d
    # this is a more "user-friendly" way to enter coordinate since in da6d the coordinates are not in degres
    #desti_lon = list(range(min_longitude, max_longitude, len(da6d[0,0,:,0,0,0])))
    #desti_lat = list(range(max_latitude, min_latitude, -len(da6d[0,0,:,0,0,0])))

    # take the closest value from the list
    #lon = min(desti_lon, key=lambda x:abs(x--user_lon))
    #lat = min(desti_lat, key=lambda x:abs(x--user_lat))

    mask_da = xr.DataArray(
        mask,
        dims=da6d.dims,
        coords=da6d.coords,
        name="anomaly_mask"
    )

    # -------- pick the nearest grid-cell by real lon/lat --------
    point_da   = da6d.sel(  dim_2=user_lon, dim_1=user_lat, method='nearest' )
    point_mask = mask_da.sel(dim_2=user_lon, dim_1=user_lat, method='nearest')

    # pick surface & variable
    raw_pixel  = point_da .isel(dim_3=depth_k, dim_4=var_idx)   # (year,month)
    pixel_mask = point_mask.isel(dim_3=depth_k, dim_4=var_idx)  # (year,month)

    # flatten to 1-D

    ts   = raw_pixel .values.flatten()
    flag = pixel_mask.values.flatten().astype(bool)

    print("flag:", flag, "   sum:", flag.sum())  # should now show exactly 2

    # build baseline
    baseline12 = raw_pixel.mean(dim='year').values
    baseline   = np.tile(baseline12, raw_pixel.sizes['year'])

    # build x & xticks (unchanged)...
    # then draw the vbar shading etc.

    n_years    = raw_pixel.sizes['year']
    x          = np.arange(1, ts.size+1)
    years      = da6d.year.values
    if periodicity == 'monthly':
        n_months   = raw_pixel.sizes['month']
    else:
        n_months   = raw_pixel.sizes['week']
    xticks     = [f"{y}-{m:02d}" for y in years for m in range(1,n_months+1)]

    # 5) Plot, with vbar for shading
    from bokeh.plotting import figure, show
    from bokeh.models     import FixedTicker
    p = figure(
        title="Raw seasonal cycle with anomalies shaded",
        x_axis_label="Time (year-month)",
        y_axis_label=f"Variable {variable[var_idx]}",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )

    # shade:
    anom_x   = x[flag]
    anom_top = ts[flag]
    anom_bot = min(ts)-0.1
    p.vbar(x=anom_x, top=anom_top, bottom=anom_bot, width=0.2,
        fill_color="red", fill_alpha=0.3, line_color=None)

    # lines:
    p.line(x, ts,       line_color="blue",  legend_label="Raw value")
    p.line(x, baseline, line_color="black", legend_label="Baseline")

    # ticks:
    tick_pos = x[::3]
    tick_lbl = xticks[::3]
    p.xaxis.ticker = FixedTicker(ticks=tick_pos)
    p.xaxis.major_label_overrides = dict(zip(tick_pos, tick_lbl))
    p.xaxis.major_label_orientation = 3.14/4
    p.legend.location = "top_left"

    show(p)

def showClimatology(climatology):
    lat_c = (min_latitude + max_latitude) / 2
    lon_c = (min_longitude + max_longitude) / 2
    
    climatology = climatology.sel(latitude = lat_c, longitude = lon_c, method="nearest")
    climatology = climatology.sortby('month')
    baseline = climatology.values.flatten()
    xticks = climatology['month']
    
    plt.plot(xticks, baseline, '--r', label = "Baseline", color = "black" )
    plt.title("baseline plot")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

def showGraph(da4d, threshold):
    plt.clf()
    
    da3d = da4d.isel(depth=0, drop=True)
    da1d = da3d.mean(dim=('latitude','longitude'))
    da1d['time'] = da1d['time'].dt.dayofyear
    
    threshold = threshold.mean(dim=('latitude','longitude'))
    mid_month_day = [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]
    daily_days = np.arange(1, 366)
    day_coord = xr.DataArray(mid_month_day, dims="month", coords={"month": np.arange(1, 13)})
    threshold = threshold.assign_coords(time = day_coord)
    threshold = threshold.drop_vars('month')
    threshold = threshold.rename({"month": "time"})
    threshold = threshold.interp(time = daily_days)

    da1d.to_netcdf("results/da1d.nc")
    threshold.to_netcdf("results/threshold.nc")

    da1d.plot(label = 'Variable')
    threshold.plot(label = 'Threshold', linestyle = '--')
    # plt.plot(xticks, var, label = "Raw Value", color = "blue")
    # plt.plot(xticks, threshold, '--r', label = "Baseline", color = "black" )
    # plt.plot(xticks, [threshold]*len(xticks), '--r', label = 'Threshold')
    plt.grid(True)
    
    plt.title(f"Anomaly Detection for ")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/anomaly_plot.png')
    plt.close()
    
def ShowGraphAnomalyv2(da6d, mask, threshold, var_index, user_lon, user_lat):

    # creation of list containing all longitude and latitude, same size as each lon/lat in da6d
    # this is a more "user-friendly" way to enter coordinate since in da6d the coordinates are not in degres
    #lon_vals = np.arange(min_longitude,max_longitude + area_resolution/2,area_resolution)
    #lat_vals = np.arange(min_latitude,max_latitude + area_resolution/2,area_resolution)

    # user_lon = float(input("Enter longitude (Array Index): "))
    # user_lat = float(input("Enter latitude  (Array Index): "))

    # take the closest value from the list
    #user_lon = np.abs(lon_vals - user_lon).argmin()
    #user_lat = np.abs(lat_vals - user_lat).argmin()

    depth_k = 0
    var_idx = var_index

    # creation of list containing all longitude and latitude, same size as each lon/lat in da6d
    # this is a more "user-friendly" way to enter coordinate since in da6d the coordinates are not in degres
    #desti_lon = list(range(min_longitude, max_longitude, len(da6d[0,0,:,0,0,0])))
    #desti_lat = list(range(max_latitude, min_latitude, -len(da6d[0,0,:,0,0,0])))

    # take the closest value from the list
    #lon = min(desti_lon, key=lambda x:abs(x--user_lon))
    #lat = min(desti_lat, key=lambda x:abs(x--user_lat))

    mask_da = xr.DataArray(
        mask,
        dims=da6d.dims,
        coords=da6d.coords,
        name="anomaly_mask"
    )

    # -------- pick the nearest grid-cell by real lon/lat --------
    point_da   = da6d.sel(  dim_2=user_lon, dim_1=user_lat, method='nearest' )
    point_mask = mask_da.sel(dim_2=user_lon, dim_1=user_lat, method='nearest')

    # pick surface & variable
    raw_pixel  = point_da.isel(dim_3=depth_k, dim_4=var_idx)   # (year,month)
    pixel_mask = point_mask.isel(dim_3=depth_k, dim_4=var_idx)  # (year,month)

    # flatten to 1-D

    ts   = raw_pixel.values.flatten()
    flag = pixel_mask.values.flatten().astype(bool)

    print("flag:", flag, "   sum:", flag.sum())  # should now show exactly 2

    # build baseline
    baseline12 = raw_pixel.mean(dim='month', skipna=True).values
    baseline   = np.tile(baseline12, raw_pixel.sizes['month'])
    ts_interp = pd.Series(ts).interpolate(method='linear').values

    # build x & xticks (unchanged)...
    # then draw the vbar shading etc.

    n_years    = raw_pixel.sizes['year']
    x          = np.arange(1, ts.size+1)
    years      = da6d.year.values
    if periodicity == 'monthly':
        n_months   = raw_pixel.sizes['month']
    else:
        n_months   = raw_pixel.sizes['week']
    xticks     = [f"{y}-{m:02d}" for y in years for m in range(1,n_months+1)]

    # 5) Plot, with vbar for shading
    plt.figure(figsize=(12,5))
    
    plt.plot(xticks, ts_interp, label = "Raw Value", color = "blue")
    plt.plot(xticks, baseline, '--r', label = "Baseline", color = "black" )
    plt.plot(xticks, [threshold]*len(xticks), '--r', label = 'Threshold')
    plt.grid(True)

    plt.title(f"Anomaly Detection for {variable[var_index]}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()