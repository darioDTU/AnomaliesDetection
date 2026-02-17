import xarray as xr

from myfunctions.AnomalyAlgorithm import AnomaliesCalculation
from myfunctions.fetcher import CopernicusFetcher
from matplotlib import pyplot as plt
import numpy as np
import copernicusmarine
import os
import calendar

# dataset = 'cmems_mod_glo_bgc_my_0.25deg_P1D-m'
dataset = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
# variables = ['chl']
variables = ['thetao']
start_year = 1995
end_year = 1996
min_latitude = -80
max_latitude = 90
min_longitude = -180
max_longitude = 179.75
# min_depth = 0.5057600140571594
min_depth = 0.49402499198913574
# max_depth= 0.5057600140571594
max_depth= 0.49402499198913574

# List to store monthly climatologies
monthly_climatologies = []
yearly_climatologies = []

for month in range(1, 13):
    for year in range(start_year, end_year + 1):
        starting_time = f"{year}/{month:02d}/01"
        last_day = calendar.monthrange(year, month)[1]
        ending_time = f"{year}/{month:02d}/{last_day:02d}"
    
        print(f"\nProcessing month {month} of year {year}...")
        xa = copernicusmarine.open_dataset(
        dataset_id = dataset,
        variables = variables,
        username = "dcannistra",
        password = ",b9UCHyV&xNzm;A",
        minimum_longitude = min_longitude,
        maximum_longitude = max_longitude,
        minimum_latitude = min_latitude,
        maximum_latitude = max_latitude,
        start_datetime = starting_time,
        end_datetime = ending_time,
        minimum_depth = min_depth,
        maximum_depth = max_depth
        )

        yearly_climatologies.append(xa)
        xa.close()
    climatology_year = xr.concat(yearly_climatologies, dim='time')
    climatology_year_mean = climatology_year.groupby('time.month').mean(dim='time', skipna=True)
    # climatology_year_mean = climatology_year_mean.assign_coords(month=month)
    yearly_climatologies = []
    monthly_climatologies.append(climatology_year_mean)

print("\nCombining all months...")
climatology = xr.concat(monthly_climatologies, dim='month')
print("\nFinal climatology dataset:")
print(climatology)
print(f"\nTotal size: {climatology.nbytes / (1024**3):.2f} GB")

# Load data into memory before saving
print("\nLoading data into memory...")
climatology = climatology.load()

# Save the complete climatology
climatology.to_netcdf(f'climatology_{dataset}_{'-'.join(variables)}_{start_year}-{end_year}_global.nc')
print(f"\nSaved to: climatology_{dataset}_{'-'.join(variables)}_{start_year}-{end_year}_global.nc")

    # Verify data structure and content
'''
    print(f"\nDataset structure:")
    print(xa)
    print(f"\nCoordinates:")
    print(f"  Time: {xa.time.values}")
    print(f"  Depth: {xa.depth.values}")
    print(f"  Latitude range: {xa.latitude.values.min():.2f} to {xa.latitude.values.max():.2f}")
    print(f"  Longitude range: {xa.longitude.values.min():.2f} to {xa.longitude.values.max():.2f}")
    
    # Check for valid (non-NaN) data
    thetao_data = xa[variable]
    total_points = thetao_data.size
    valid_points = (~thetao_data.isnull()).sum().values
    print(f"\nData check:")
    print(f"  Total data points: {total_points}")
    print(f"  Valid (non-NaN) points: {valid_points}")
    print(f"  Percentage valid: {(valid_points/total_points)*100:.2f}%")
    
    # Find a location with valid data
    if valid_points > 0:
        # Get first timestep and find valid point
        first_time = thetao_data.isel(time=0, depth=0)
        valid_mask = ~first_time.isnull()
        
        if valid_mask.any():
            lat_idx, lon_idx = np.where(valid_mask.values)
            test_lat = float(first_time.latitude[lat_idx[0]].values)
            test_lon = float(first_time.longitude[lon_idx[0]].values)
            
            # Test selection at this valid point
            test_value = xa[variable].sel(
                depth=xa.depth.values[0],
                latitude=test_lat,
                longitude=test_lon,
                time=xa.time.values[0],
                method="nearest"
            ).values
            
            print(f"\nExample valid data point:")
            print(f"  Location: ({test_lat:.2f}, {test_lon:.2f})")
            print(f"  Value at first timestep: {test_value:.2f}°C")
    
    # Average over the time dimension to get a single monthly climatology
    db_averaged = xa.mean(dim='time')
    
    # Add month coordinate
    
    
    # Calculate size
    size_gb = db_averaged.nbytes / (1024**3)
    print(f"Month {month} averaged dataset size: {size_gb:.2f} GB")
    
    monthly_climatologies.append(db_averaged)
    '''

# # Plot the climatology
# plt.figure(figsize=(10, 6))
# plt.plot(clim_loaded['month'], clim_loaded, marker='o', linewidth=2, markersize=8, label='Thetao Climatology')
# plt.xlabel('Month', fontsize=12)
# plt.ylabel('Temperature (°C)', fontsize=12)
# plt.title(f'Monthly Climatology (1993-2003) at ({clim_loaded.latitude.values:.2f}°, {clim_loaded.longitude.values:.2f}°)', fontsize=14)
# plt.xticks(range(1, 13))
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.savefig("clim_thetao.png", dpi=150)
# print("\nPlot saved to clim_thetao.png")