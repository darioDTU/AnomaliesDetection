from datetime import datetime

import xarray as xr
import copernicusmarine

class CopernicusFetcher:
    
    '''Download data from Copernicus databases.'''

    dataset_id : str
    starting_time : str
    ending_time : str
    variable : str
    minimum_longitude : float
    maximum_longitude : float
    minimum_latitude : float
    maximum_latitude : float
    minimum_depth : float
    maximum_depth : float 
    
    def __init__(self,
                dataset_id : str, 
                starting_time : str, 
                ending_time : str, 
                variable : str, 
                minimum_longitude : float, 
                maximum_longitude : float,
                minimum_latitude : float, 
                maximum_latitude : float,
                minimum_depth : float,
                maximum_depth : float) -> None:
        
        self.dataset_id = dataset_id
        self.starting_time = starting_time
        self.ending_time = ending_time
        self.variable = variable
        self.minimum_longitude = minimum_longitude
        self.maximum_longitude = maximum_longitude
        self.minimum_latitude = minimum_latitude
        self.maximum_latitude = maximum_latitude
        self.minimum_depth = minimum_depth
        self.maximum_depth = maximum_depth

    def __set_superficial_depth(self) -> None:

        '''Set superficial depth for climatology data fetching.'''

        superficial_depth = 0.49402499198913574
        self.maximum_depth = superficial_depth
        self.minimum_depth = superficial_depth

    def __set_climatology_time(self) -> None:

        '''Set time range for climatology data fetching.'''

        self.starting_time = "2010/01/01"
        self.ending_time = datetime.now().strftime("%Y/%m/%d")
        self.dataset_id = self.dataset_id.replace("D", "M")

    def fetch_temperature(self, climatology : bool = False) -> xr.Dataset:

        '''Fetch temperature data from Copernicus database.'''
        if climatology:
            self.__set_climatology_time()

        if self.minimum_depth < 0:
            self.__set_superficial_depth()

        ds = copernicusmarine.open_dataset(
            dataset_id = self.dataset_id,
            variables = [self.variable],
            username = "dcannistra",
            password = ",b9UCHyV&xNzm;A",
            minimum_longitude = self.minimum_longitude,
            maximum_longitude = self.maximum_longitude,
            minimum_latitude = self.minimum_latitude,
            maximum_latitude = self.maximum_latitude,
            start_datetime = self.starting_time,
            end_datetime = self.ending_time,
            minimum_depth = self.minimum_depth,
            maximum_depth = self.maximum_depth
            )
        
        return ds

