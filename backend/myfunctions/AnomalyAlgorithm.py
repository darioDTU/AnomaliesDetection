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


class AnomaliesCalculation:

    min_latitude : float
    min_longitude : float
    starting_time : str
    dataset : str
    percentile : int
    variable : str

    def __init__(self, min_latitude: float, min_longitude: float, starting_time : str,dataset: str, percentile : int, variable: str) -> None:
        self.min_latitude = min_latitude
        self.min_longitude = min_longitude
        self.starting_time = starting_time
        self.dataset = dataset
        self.percentile = percentile
        self.variable = variable
    def ClimatologyCalculation(self, baseline, dataset, variable) -> None:
        # compute the mean across the 'time' dimension
        climatology = baseline.groupby("time.month").mean("time", skipna=True)
        # climatology = climatology.squeeze()
        climatology = climatology[variable]
        climatology.to_netcdf(f"climatology_{self.dataset}_{self.min_latitude}_{self.min_longitude}_{self.variable}.nc")

    def __get_year(self):
        return self.starting_time.split("/")[-1]

    def __compute_variable1d(self, variable4d):

        variable3d = variable4d.mean(dim='depth', skipna=True)
        variable1d = variable3d.mean(dim=('latitude','longitude'))

        return variable1d

    def __compute_da1d(self, da4d):
        
        da1d = self.__compute_variable1d(da4d)
        da1d['time'] = da1d['time'].dt.dayofyear
        daily_days = np.arange(1, 367)
        da1d = da1d.interp(coords={"time": daily_days})
        
        return da1d
    def __interpolate_variable(self, variable1d):
        
        daily_days = np.arange(1, 367)
        mid_month_day = [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]
        extended_days = np.concatenate(([mid_month_day[-1] - 365], mid_month_day, [mid_month_day[0] + 365]))
        extended_values = np.concatenate(([variable1d[-1]], variable1d, [variable1d[0]]))
        
        variable_extended = xr.DataArray(
            extended_values,
            dims = "time",
            coords={"time":extended_days}
        )
        variable_extended = variable_extended.interp(coords={"time": daily_days})
        
        return variable_extended
    def AnomalyDetection(self, da4d, climatology):
        # clim0 = climatology.isel(year=0, drop=True)½
        # climatology = climatology.assign_coords(month=climatology['month'].astype(int))
        # climatology = climatology.to_array(dim='var')
        # data = data.to_array(dim='var')
        data_grouped = da4d.groupby("time.month").mean()
        deviation = data_grouped - climatology
        vals = deviation.load()  
        vals_array = abs(vals.data)
        valid = vals_array[np.isfinite(vals_array)]
        if valid.size == 0:
            raise ValueError("No finite deviations found! Check your data/climatology overlap.")
        threshold_value = float(np.nanpercentile(valid, self.percentile))
        mask = deviation >= threshold_value
        anomalies = deviation.where(mask)

        threshold_array = climatology + threshold_value
        return [threshold_value], threshold_array

    #TODO modify percentile
    def POT(self, X, q=0.01, t=None, t_pct=90/100):
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

    def __DSPOT(self, X, d, n=5000, q=0.01):
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
        zq, t = self.POT(cal, q, t_pct=self.percentile/100)

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

    def ProcessAnomalies(self, da4d, climatology, res):
        
        if res == 0:
            threshold_value, threshold_array = self.AnomalyDetection(da4d, climatology)

        elif res == 1:
            # climatology = climatology.to_array(dim='var')
            threshold_value = []
            for climatology_per_month in climatology:
                threshold_value_monthly, t = self.POT(climatology_per_month)
                threshold_value.append(threshold_value_monthly)

            months = np.arange(1, 13)
            
            threshold_array = xr.DataArray(
                    threshold_value,
                    dims=["month"],
                    coords={"month": months},
                    name="pot_threshold"
                )
        else:
            _, threshold_value, _ = self.__DSPOT(X=climatology, d=0)
            threshold_array = None
        return threshold_value, threshold_array

    def DetectAnomalies(self, da6d, threshold):
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

    def showClimatology(self, climatology):
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

    def showGraph(self, da4d, threshold, threshold_value, variable_name):
        plt.clf()
        
        da1d = self.__compute_da1d(da4d)
        threshold1d = self.__compute_variable1d(threshold)

        threshold = threshold.mean(dim=('latitude','longitude'))
        threshold_extended = self.__interpolate_variable(threshold1d)
        threshold_value = np.array(threshold_value)
        climatology = threshold_extended - threshold_value
                               
        da1d.to_netcdf("results/da1d.nc")
        threshold_extended.to_netcdf("results/threshold.nc")

        plt.plot(da1d['time'], da1d, label = 'Variable')
        plt.plot(climatology['time'], climatology.values, label='Climatology', color='black', linewidth=2, alpha=0.3)
        plt.plot(threshold_extended['time'], threshold_extended.values, label='Threshold', linestyle='--')
        plt.fill_between(da1d['time'], da1d, threshold_extended, where=(da1d > threshold_extended), color='red', alpha=0.3, label='Anomaly 95th Percentile')
        plt.grid(True)
        
        plt.title(f"Anomaly Detection for {self.__get_year()}")
        plt.xlabel("Time [Day of Year]")
        plt.ylabel(variable_name)
        plt.legend()
        plt.tight_layout()
        plt.savefig('results/plot.png')
        plt.close()
        
    def showGraphPOT(self, da4d, threshold, climatology4d, variable_name):
        plt.clf()
        
        da1d = self.__compute_da1d(da4d)
        
        climatology1d = self.__compute_variable1d(climatology4d)
        
        threshold_extended = self.__interpolate_variable(threshold)
        climatology = self.__interpolate_variable(climatology1d)
                               
        da1d.to_netcdf("results/da1d.nc")
        threshold_extended.to_netcdf("results/threshold.nc")

        plt.plot(da1d['time'], da1d, label = 'Variable')
        plt.plot(climatology['time'], climatology.values, label='Climatology', color='black', linewidth=2, alpha=0.3)
        plt.plot(threshold_extended['time'], threshold_extended.values, label='Threshold (POT)', linestyle='--')
        plt.fill_between(da1d['time'], da1d, threshold_extended, where=(da1d > threshold_extended), color='red', alpha=0.3, label='Anomalies')
        plt.grid(True)
        
        plt.title(f"Anomaly Detection for {self.__get_year()}")
        plt.xlabel("Time [Day of Year]")
        plt.ylabel(variable_name)
        plt.legend()
        plt.tight_layout()
        plt.savefig('results/plot.png')
        plt.close()