import time
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from myfunctions.AlgoPy import AlgoPy

def PlotHeatmap(starting_time, ending_time):
    """
    Plots a heatmap of platform availability globally.

    Parameters:
        data (DataFrame): Data containing 'PLATFORM_NUMBER', 'LATITUDE', 'LONGITUDE'.
    """
 # Use numpy to define longitude and latitude ranges
    longs1 = range(-180, 180, 15)
    lats1 = [-30, -90]
    longs2 = range(-180, 180, 15)
    lats2 = [30, -30]
    longs3 = range(-180, 180, 15)
    lats3 = [90, 30]

    # Initialize your algorithm or system setup
    data_points = []

    # Initialize your algorithm or system setup
    algo = AlgoPy()
    algo.starting_time = starting_time
    algo.ending_time = ending_time
    algo.max_depth = 10
    # Process each geographic segment
    for long_range, lat_range in zip([longs1, longs2, longs3], [lats1, lats2, lats3]):
        for i in range(len(long_range) - 1):
            for y in range(len(lat_range) - 1):
                algo.min_longitude = long_range[i]
                algo.max_longitude = long_range[i+1]
                algo.min_latitude = lat_range[y+1]
                algo.max_latitude = lat_range[y]

                # Extract and store data
                try:
                    result, column_index = algo.ExtractData(heatmap=1)
                    if isinstance(result, np.ndarray) and result.size > 0:
                        latitudes = result[:, column_index['LATITUDE']]
                        longitudes = result[:, column_index['LONGITUDE']]
                        data_points.append((longitudes, latitudes))
                except Exception as e:
                    print(f"Error extracting data: {e}")
                time.sleep(2)
    if not data_points:
        print("No data available for heatmap.")
        return
    # Aggregate all longitude and latitude points
    all_longitudes = np.concatenate([item[0] for item in data_points])
    all_latitudes = np.concatenate([item[1] for item in data_points])
    
    lat_bins = np.linspace(-90, 90, 13)
    lon_bins = np.linspace(-180, 180, 25)
    
    # After computing the histogram:
    heatmap, xedges, yedges = np.histogram2d(all_longitudes, all_latitudes, bins=[lon_bins, lat_bins])
    # Compute the centers of bins for longitudes and latitudes:
    lon_centers = 0.5 * (xedges[:-1] + xedges[1:])
    lat_centers = 0.5 * (yedges[:-1] + yedges[1:])
    
    # Create a meshgrid so that every bin center is assigned to a coordinate:
    X, Y = np.meshgrid(lon_centers, lat_centers, indexing='ij')
    # Now both X and Y will have shape (len(lon_centers), len(lat_centers))
    
    # Plot setup
    fig, ax = plt.subplots(figsize=(24, 12), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    # Use scatter plot with the grid flattened; they must be the same size:
    ax.scatter(X.flatten(), Y.flatten(), c=heatmap.ravel(), s=heatmap.ravel() * 0.06, cmap='viridis', alpha=0.75, edgecolor='none')
    plt.title("Global Heatmap of Platform Availability")
    plt.show()