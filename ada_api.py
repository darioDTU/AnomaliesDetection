import base64
import os
from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
from pydantic import BaseModel
import xarray as xr
# from dask.distributed import Client
from myfunctions.AnomalyAlgorithm import ClimatologyCalculation, DetectAnomalies, ProcessAnomalies, showClimatology, ShowGraphAnomalyv2, ShowPixelAnomalies, showGraph
from myfunctions.coordinates_values import *
from myfunctions.AlgoPy import AlgoPy
from myfunctions.fetcher import CopernicusFetcher

xr.set_options(keep_attrs=True, display_expand_data=False)
# client = Client(n_workers=2, threads_per_worker=2, memory_limit='1GB')
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

    def get_stats(self, da1d, threshold):
        
        '''Get statistics of the anomalies.'''
        
        anomaly = da1d - threshold
        anomaly = anomaly.where(anomaly > 0)

        max_anom = float(anomaly.max(skipna=True).values) if anomaly.size else float('nan')
        t_max = anomaly.where(anomaly == max_anom, drop=True)['time'].values
        max_anom_real = float(da1d.sel(time = t_max).values)
        intensity = float(anomaly.sum().values)
        mean = float(da1d.mean().values)

        max_anom = round(max_anom, 3)
        max_anom_real = round(max_anom_real, 3)
        intensity = round(intensity, 3)
        mean = round(mean, 3)
        
        return max_anom, max_anom_real, intensity, mean
    
    def load_job_file(self, suffix: str) -> xr.DataArray:
        
        path = f"{suffix}"
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"File {suffix} not found.")
        print('loading file', path)
        return xr.open_dataarray(path)

class PipelineParams(BaseModel):
    dataset: str
    starting_time: str

app = FastAPI()

origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # or ["*"] for dev
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,    # set True only if you use cookies/auth
)

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

@app.post("/run_pipeline_anomalies/classic")
async def run_pipeline_classic(params : PipelineParams):
    
    dataset = params.dataset
    starting_time = params.starting_time

    output = CopernicusFetcher().fetch_temperature(dataset, starting_time)
    climatology = xr.open_dataarray("climatology.nc")
    da4d = output['thetao']
    threshold = ProcessAnomalies(da4d, climatology, 0)
    showGraph(da4d, threshold)
    return {
        "Status": "ok"}

@app.get("/show_image")
async def show_image():
    return FileResponse("results/anomaly_plot.png", media_type="image/png")

@app.get("/get_stats")
async def get_stats():
    try:
        threshold = APIHelper().load_job_file("results/threshold.nc")
        da1d = APIHelper().load_job_file("results/da1d.nc")
        max_anom, max_anom_real, intensity, mean = APIHelper().get_stats(da1d, threshold)
    except Exception as e:
        print(f'Error in get_stats: {e}')
        return {}

    return {
        "Status": "ok",
        "Statistics": {
            "Max Anomalies Gap [°C]": max_anom,
            "Max Anomalies [°C]": max_anom_real,
            "Intensity [°C/Days]": intensity,
            "Mean Temperature [°C]": mean
        }
    }

# output = xr.load_dataarray("output_array.nc")
# baseline = CopernicusFetcher().fetch_temperature(dataset_baseline, starting_time_bs)
# output = CopernicusFetcher().fetch_temperature(dataset, starting_time)
# da4d, climatology = ClimatologyCalculation(baseline, output)
# da4d = output['thetao']
# climatology = xr.open_dataarray("climatology.nc")
# threshold = ProcessAnomalies(da4d, climatology, 0)
# showGraph(da4d, threshold)
# mask, coords = DetectAnomalies(da6d, threshold)
# ShowPixelAnomalies(mask, 0)
# ShowGraphAnomalyv2(da6d, mask, threshold, 0, user_lon = 30, user_lat = 25)

dataset = 'cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m'
starting_time = '01/01/2023'
output = CopernicusFetcher().fetch_temperature(dataset, starting_time)
climatology = xr.open_dataarray("climatology.nc")
da4d = output['thetao']
threshold = ProcessAnomalies(da4d, climatology, 0)
showGraph(da4d, threshold)