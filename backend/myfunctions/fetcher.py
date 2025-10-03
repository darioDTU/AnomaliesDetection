from datetime import datetime
from myfunctions.coordinates_values import *

import xarray as xr
import copernicusmarine

class CopernicusFetcher:
    
    '''Download data from Copernicus databases.'''
    
    def __init__(self) -> None:
        self.min_longitude = min_longitude
        self.max_longitude = max_longitude
        self.min_latitude = min_latitude
        self.max_latitude = max_latitude
        self.starting_time = starting_time
        self.ending_time = ending_time
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.dataset = dataset_argo
    
    def fetch_temperature(self, 
                          dataset_id : str, 
                          starting_time : str, 
                          ending_time : str, 
                          variable : str, 
                          minimum_longitude : float, 
                          maximum_longitude : float,
                          minimum_latitude : float, 
                          maximum_latitude : float,
                          climatology : bool = False) -> xr.Dataset:

        '''Fetch temperature data from Copernicus database.'''
        if climatology:
            starting_time = "2010/01/01"
            ending_time = datetime.now().strftime("%Y/%m/%d")
            dataset_id = dataset_id.replace("D", "M") 

        ds = copernicusmarine.open_dataset(
            dataset_id = dataset_id,
            variables = [variable],
            username = "dcannistra",
            password = ",b9UCHyV&xNzm;A",
            minimum_longitude = minimum_longitude,
            maximum_longitude = maximum_longitude,
            minimum_latitude = minimum_latitude,
            maximum_latitude = maximum_latitude,
            start_datetime = starting_time,
            end_datetime = ending_time,
            minimum_depth = self.min_depth,
            maximum_depth = self.max_depth
            )
        
        return ds

    def fetch_salinity(self):
        return True
