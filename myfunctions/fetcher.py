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
    
    def __get_ending_date(self, starting_time : str):

        '''Get the year from a date.'''
        
        year = starting_time.split('/')[-1]
        return f"31/12/{year}"
    
    def fetch_temperature(self, dataset_id, starting_time) -> xr.Dataset:
        
        ds = copernicusmarine.open_dataset(
            dataset_id = dataset_id,
            variables = ["thetao"],
            username = "dcannistra",
            password = ",b9UCHyV&xNzm;A",
            minimum_longitude = self.min_longitude,
            maximum_longitude = self.max_longitude,
            minimum_latitude = self.min_latitude,
            maximum_latitude = self.max_latitude,
            start_datetime = starting_time,
            end_datetime = self.__get_ending_date(starting_time),
            minimum_depth = self.min_depth,
            maximum_depth = self.max_depth
            )
        
        return ds

    def fetch_salinity(self):
        return True
