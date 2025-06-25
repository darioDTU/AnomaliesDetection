import sys
import argopy
import datetime
import numpy as np
import xarray as xr
import pandas as pd
from argopy import DataFetcher
import matplotlib.pyplot as plt
from argopy.plot import scatter_map
from myfunctions.Subplot3 import Subplot3
from myfunctions.coordinates_values import *
from myfunctions.ProcessWeek import ProcessWeek
from myfunctions.SentenceToString import SentenceToString

class AlgoPy():
    fetcher = []
    pd_fetcher = []
    columns_index = []

    def __init__(self):
        """
        Initialisation of datas declared previously in the class
        """
        self.min_longitude = min_longitude
        self.max_longitude = max_longitude
        self.min_latitude = min_latitude
        self.max_latitude = max_latitude
        self.starting_time = starting_time
        self.ending_time = ending_time
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.dataset = dataset

    def ExtractData(self, heatmap=0):
        """
        Extract a DataFrame from initiated datas

        In :
            -> Minimum longitude
            -> Maximum longitude
            -> Minimum latitude
            -> Maximum latitude
            -> Starting time
            -> Ending time
            -> Dataset
                --> 3 possible choices : erddap (default), gdac, argovis
            -> Minimum Depth
            -> Maximum Depth
        Out :
            DataFrame containing datas for the specified inputs
        """
        try:
            if dataset == 'erddap':
                argopy.set_options(mode='expert', ds='bgc')
                fetcher = DataFetcher(src=self.dataset, mode='expert').region([self.min_longitude, self.max_longitude, self.min_latitude, self.max_latitude, self.min_depth, self.max_depth, self.starting_time, self.ending_time]).to_xarray()
            else:
                argopy.set_options(mode='standard', ds='phy')
                fetcher = DataFetcher(src=self.dataset, mode='standard').region([self.min_longitude, self.max_longitude, self.min_latitude, self.max_latitude, self.min_depth, self.max_depth, self.starting_time, self.ending_time]).to_xarray()
            fetcher = fetcher.to_dataframe()
            self.pd_fetcher = fetcher
            # if the processing is for a heatmap, delete all data to only keep what we need
            if heatmap == 1:
                del fetcher
                fetcher = pd.DataFrame()
            else:
                fetcher = pd.concat([fetcher[i] for i in variable], axis = 1)
            fetcher = pd.concat([fetcher, self.pd_fetcher['PLATFORM_NUMBER'], self.pd_fetcher['LATITUDE'], self.pd_fetcher['LONGITUDE'], self.pd_fetcher['PRES'], self.pd_fetcher['TIME']], axis=1)
            columns_index = {col: idx for idx, col in enumerate(fetcher.columns)}
            fetcher = fetcher.values
            self.columns_index = columns_index
            self.fetcher = fetcher
            return fetcher, columns_index
        except ValueError as e:
            words_in_error = SentenceToString(e)
            if 'lat_max' in words_in_error:
                print('Maximum Latitude should be superior to minimum latitude')
            elif 'lon_max' in words_in_error:
                print('Maximum longitude should be superior to minimum longitude')
            elif 'pres_max' in words_in_error:
                print('Maximum depth should be superior to minimum depth')
            elif 'datetim_max' in words_in_error:
                print('Ending time should be superior to starting time')
            elif "'src'" in words_in_error:
                print('Please check if the selected dataset exist or spelling is good')
            elif "'mode'" in words_in_error:
                print('Please check if selected mode exist or spelling is good')
            else:
                print(f"Following ValueError as occured '''{e.args[0]}'''")
            return
        except FileNotFoundError:
            print('No datas available in the selected zone, zone is too wide or server issue') # if this error occurs, there's a lot of possibility so try to search the code error on the internet for example "error 500 erddap"
            sys.exit()

    def PlotRegion(self):
        """
        Plot a region according to the specified inputs

        In :
            -> Result of ExtractData function
        Out :
            Plot the data of the selected region with each platform numbers
        """
        scatter_map(self.pd_fetcher, hue='PLATFORM_NUMBER',x='LONGITUDE', y='LATITUDE', set_global=True)
        del self.pd_fetcher
        plt.show()

    def PlotDataDepth(self, platform_number):
        """
        Plot graphs of differents datas according to Depth on a selected sensor from function PlotRegion

        In :
            -> PLATFORM_NUMBER
        Out :
            Plot of datas according to Depth
        """
        platform_number_column = self.columns_index['PLATFORM_NUMBER']
        latitude_index = self.columns_index['LATITUDE']
        longitude_index = self.columns_index['LONGITUDE']

        if platform_number not in self.fetcher[:, platform_number_column]:
            print('The selected platform number does not exist in the selected zone')
            return
        mask = self.fetcher[:, platform_number_column] == platform_number
        platform = self.fetcher[mask]
        unique_latitude_values = np.unique(platform[:, latitude_index])

        if self.dataset == 'erddap':
            all_datas = ['CHLA', 'TEMP', 'PSAL', 'DOXY', 'NITRATE', 'BBP700', 'DOWN_IRRADIANCE412', 'DOWN_IRRADIANCE443', 'PH_IN_SITU_TOTAL', 'CDOM']
        else:
            all_datas = ['TEMP', 'PSAL']
        datas_available = []

        for datas in all_datas:
            numeric_column = platform[:, self.columns_index[datas]].astype(float)
            platform_bis = platform[~np.isnan(numeric_column)]
            if len(platform_bis) != 0:
                datas_available.append(datas)

        if len(datas_available) > 1:
            print('Multiple kind of datas are available, please select the one you want to see:')
            for i in range(len(datas_available)):
                if datas_available[i] == 'CHLA':
                    print(f"{[i]} for Chlorophyll-a")
                elif datas_available[i] == 'TEMP':
                    print(f"{[i]} for Temperature")
                elif datas_available[i] == 'PSAL':
                    print(f"{[i]} for Salinity")
                elif datas_available[i] == 'DOXY':
                    print(f"{[i]} for Oxygen")
                elif datas_available[i] == 'NITRATE':
                    print(f"{[i]} for Nitrate")
                elif datas_available[i] == 'BBP700':
                    print(f"{[i]} for Backscatter")
                elif datas_available[i] == 'DOWN_IRRADIANCE412':
                    print(f"{[i]} for Downwelling Irradiance 412nm")
                elif datas_available[i] == 'DOWN_IRRADIANCE443':
                    print(f"{[i]} for Downwelling Irradiance 443nm")
                elif datas_available[i] == 'PH_IN_SITU_TOTAL':
                    print(f"{[i]} for PH in situ")
                elif datas_available[i] == 'CDOM':
                    print(f"{[i]} for Colored Dissolved Organic Matter")
            selected_values = input(f"Enter indices (space-separated) or type '{i+1}' for all: ").strip()

            if selected_values == str(i+1):  # Keep all values
                pass  # No filtering needed
            else:
                try:
                    # Convert input to list of integers
                    list_of_values = list(map(int, selected_values.split(' ')))
                    datas_available = np.array(datas_available)
                    datas_available = datas_available[list_of_values]
                except (ValueError, IndexError):
                    print("Invalid input! Please enter valid indices separated by spaces.")

        if len(unique_latitude_values) > 1:
            print('More than one coordinate is available for this platform number, please select the one you want to see:')

            for i, latitude in enumerate(unique_latitude_values):
                # Find entries that match the current latitude
                latitude_mask = platform[:, latitude_index] == latitude
                matching_longitudes = platform[latitude_mask, longitude_index]

                # Check if there are any matching longitudes
                if matching_longitudes.size > 0:
                    # Safely take the first longitude
                    longitude = matching_longitudes[0]
                    print(f"[{i}] Latitude: {latitude}; Longitude: {longitude}")
                else:
                    # This branch might be executed if there's an unexpected data inconsistency
                    print(f"Debug: No matching longitudes found for Latitude: {latitude}")

            selected_values = input(f"Enter indices (space-separated) or type '{i+1}' for all: ").strip()

            if selected_values == str(i+1):  # Keep all values
                pass  # No filtering needed
            else:
                try:
                    # Convert input to list of integers
                    list_of_values = list(map(int, selected_values.split(' ')))
                    # Keep only selected rows
                    unique_latitude_values = unique_latitude_values[list_of_values]
                except (ValueError, IndexError):
                    print("Invalid input! Please enter valid indices separated by spaces.")

        Subplot3(platform, unique_latitude_values,datas_available, self.columns_index)

    def RegionRegriding(self, area_resolution, depth_resolution, periodicity):
        """
        Vectorized regridding of selected zone with defined resolutions and mean aggregation,
        preserving per-month or per-week extraction, returning numpy arrays ready for interpolation.

        In:
            -> area_resolution (deg), depth_resolution (units), periodicity: 'monthly' or 'weekly'
        Out:
            -> output: np.ndarray of shape (T, N_lon, N_lat, N_depth, N_vars)
               grid_edges: dict with 'lat', 'lon', 'depth' edges arrays
               var_list: list of variable names in last axis order
               time_bins: list of month names or week numbers in time axis order
        """
        # Define grid edges
        lat_edges = np.arange(self.min_latitude, self.max_latitude + area_resolution, area_resolution)
        lon_edges = np.arange(self.min_longitude, self.max_longitude + area_resolution, area_resolution)
        depth_edges = np.arange(self.min_depth, self.max_depth + depth_resolution, depth_resolution)
        n_lat = len(lat_edges) - 1
        n_lon = len(lon_edges) - 1
        n_depth = len(depth_edges) - 1

        # Determine intervals (year from start)
        years = all_years

        intervals = []

        for i in range(len(years)):
            year = years[i]
            if periodicity == 'monthly':
                for m in range(1, 13):
                    start = f"{year}-{m:02d}-01"
                    end = (datetime.date(year, m % 12 + 1, 1) - datetime.timedelta(days=1)) if m < 12 else datetime.date(year, 12, 31)
                    label = datetime.date(year, m, 1).strftime('%B')
                    intervals.append((start, end.strftime('%Y-%m-%d'), label))
            else:
                for w in range(1, 53):
                    days = ProcessWeek(year, w)
                    start = days[0].strftime('%Y-%m-%d')
                    end = days[-1].strftime('%Y-%m-%d')
                    label = w
                    intervals.append((start, end, label))
            T = len(intervals)

        # Select variable list
        var_list = ['TEMP','PSAL'] if self.dataset != 'erddap' else [
            'CHLA','TEMP','PSAL','DOXY','NITRATE','BBP700',
            'DOWN_IRRADIANCE412','DOWN_IRRADIANCE443','PH_IN_SITU_TOTAL','CDOM'
        ]
        N_vars = len(var_list)

        # Prepare output array with NaNs
        output = np.full((T, n_lon, n_lat, n_depth, N_vars), np.nan, dtype=float)
        time_bins = []

        # Loop intervals, fetch and bin
        for ti, (start, end, label) in enumerate(intervals):
            self.starting_time = start
            self.ending_time = end
            # Attempt to extract data
            res = self.ExtractData()
            if res is None:
                # No data or an error occurred, skip this interval
                time_bins.append(label)
                continue
            fetcher, cols = res
            time_bins.append(label)
            if fetcher.size == 0:
                continue

            # compute indices
            lat_idx = np.digitize(fetcher[:, cols['LATITUDE']], lat_edges) - 1
            lon_idx = np.digitize(fetcher[:, cols['LONGITUDE']], lon_edges) - 1
            dep_idx = np.digitize(fetcher[:, cols['PRES']], depth_edges) - 1

            # mask valid
            valid = (
                (lat_idx >= 0) & (lat_idx < n_lat) &
                (lon_idx >= 0) & (lon_idx < n_lon) &
                (dep_idx >= 0) & (dep_idx < n_depth)
            )
            if not np.any(valid):
                continue

            idx_lat = lat_idx[valid]
            idx_lon = lon_idx[valid]
            idx_dep = dep_idx[valid]
            data = fetcher[valid]

            # flatten cell index
            cell_flat = idx_lon * (n_lat * n_depth) + idx_lat * n_depth + idx_dep

            # aggregate each variable with NaNs for empty cells
            for vi, var in enumerate(var_list):
                vals = data[:, cols[var]].astype(float)
                sums = np.bincount(cell_flat, weights=vals, minlength=n_lon*n_lat*n_depth)
                counts = np.bincount(cell_flat, minlength=n_lon*n_lat*n_depth)
                means = np.full_like(sums, np.nan, dtype=float)
                nonzero = counts > 0
                means[nonzero] = sums[nonzero] / counts[nonzero]
                output[ti, :, :, :, vi] = means.reshape((n_lon, n_lat, n_depth))
            output = xr.DataArray(output)

        # Return numpy outputs
        grid_edges = {'lat': lat_edges, 'lon': lon_edges, 'depth': depth_edges}
        return output
    
