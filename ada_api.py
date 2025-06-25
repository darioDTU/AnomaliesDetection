from fastapi import Depends, FastAPI, UploadFile
import pandas as pd
import xarray as xr
from myfunctions.AnomalyAlgorithm import ClimatologyCalculation, DetectAnomalies, ProcessAnomalies
from myfunctions.coordinates_values import *
from myfunctions.AlgoPy import AlgoPy

class APIHelper:
    
    '''API Helper class.'''
    
    coords_list : list[dict]
    
    def get_coordinate_list(self, da6d, coords): 
        
        '''Get coordinates list of the anomalies.'''
        
        coords_list = []
        for (y,t,lo,la,dep,v) in coords:
            coords_list.append({
            # 'year':  da6d.coords['year' ] .values[y].item(),
            # 'month':  da6d.coords['month' ] .values[t].item(),
            'lon':   da6d.coords['dim_1'].values[lo].item(),
            'lat':   da6d.coords['dim_2'] .values[la].item()
            # 'depth': da6d.coords['dim_3'].values[dep].item(),
            # 'var':   da6d.coords['dim_4'  ].values[v]
            })
            
            return coords_list

app = FastAPI()

@app.get("/")
async def root():
    return {"message" : "Server online"}

@app.get("/run_pipelin_anomalies/pot")
async def run_pipeline_pot():
    output = xr.load_dataarray("output_array.nc")
    da6d, climatology = ClimatologyCalculation(output)
    threshold = ProcessAnomalies(da6d, climatology, 0)
    mask, coords = DetectAnomalies(da6d, threshold)
    coords_anomalies = APIHelper().get_coordinate_list(da6d = da6d, coords = coords)
    coords_list = []
    for (y,t,lo,la,dep,v) in coords:
        coords_list.append({
        # 'year':  da6d.coords['year' ] .values[y].item(),
        # 'month':  da6d.coords['month' ] .values[t].item(),
        'lon':   da6d.coords['dim_1'].values[lo].item(),
        'lat':   da6d.coords['dim_2'] .values[la].item()
        # 'depth': da6d.coords['dim_3'].values[dep].item(),
        # 'var':   da6d.coords['dim_4'  ].values[v]
        })
    print(coords_anomalies)
    return {
        "Status": "ok", 
        "Anomalies": coords.shape[0]}

@app.get("/run_pipelin_anomalies/classic")
async def run_pipeline_classic():
    output = xr.load_dataarray("output_array.nc")
    da6d, climatology = ClimatologyCalculation(output)
    threshold = ProcessAnomalies(da6d, climatology, 0)
    mask, coords = DetectAnomalies(da6d, threshold)
    coords_anomalies = APIHelper().get_coordinate_list(da6d = da6d, coords = coords)
    return {
        "Status": "ok", 
        "Anomalies": coords.shape[0]}
