import base64
import os
import time
from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
from pydantic import BaseModel
import xarray as xr
# from dask.distributed import Client
from myfunctions.AnomalyAlgorithm import AnomaliesCalculation
from myfunctions.coordinates_values import *
from myfunctions.AlgoPy import AlgoPy
from myfunctions.fetcher import CopernicusFetcher

xr.set_options(keep_attrs=True, display_expand_data=False)

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
    latitude: float
    longitude: float
    starting_time: str
    variable : str

app = FastAPI(root_path="/api")

app.state.db_path = None
app.state.zq_pot = None

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
## Uncomment this line if you want to run test locally without a proxy
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

@app.post("/run_pipeline_anomalies/pot")
async def run_pipeline_pot(params : PipelineParams):
    dataset = params.dataset
    starting_time = f"01/01/{params.starting_time}"
    ending_time = f"31/12/{params.starting_time}"
    variable = params.variable.split('-')[0]
    variable_name = params.variable.split('-')[-1]
    min_latitude = params.latitude
    max_latitude = min_latitude + 5
    min_longitude = params.longitude
    max_longitude = min_longitude + 5

    anomalies_class = AnomaliesCalculation(min_latitude, min_longitude, starting_time,dataset, 95, variable)

    output = CopernicusFetcher().fetch_temperature(
                                        dataset_id=dataset, 
                                        starting_time=starting_time,
                                        ending_time=ending_time, 
                                        variable=variable,
                                        minimum_latitude=min_latitude,
                                        maximum_latitude=max_latitude,
                                        minimum_longitude=min_longitude,
                                        maximum_longitude=max_longitude)

    climatology_path = f"climatology_{dataset}_{min_latitude}_{min_longitude}_{variable}.nc"
    if not os.path.exists(climatology_path):
        if app.state.db_path != None:
            os.remove(app.state.db_path)
        
        app.state.db_path = climatology_path
        baseline = CopernicusFetcher().fetch_temperature(
            dataset_id=dataset,
            starting_time=starting_time,
            ending_time=ending_time,
            variable=variable,
            minimum_latitude=min_latitude,
            maximum_latitude=max_latitude,
            minimum_longitude=min_longitude,
            maximum_longitude=max_longitude,
            climatology=True
        )
        anomalies_class.ClimatologyCalculation(baseline, output, variable)
        threshold_value, threshold_array = anomalies_class.POT(baseline[variable])
        print(threshold_value)
        app.state.zq_pot = threshold_value

    climatology = xr.open_dataarray(climatology_path)
    da4d = output[variable]
    threshold_value = app.state.zq_pot
    anomalies_class.showGraph_scalar(da4d, threshold_value, climatology, variable_name)
    return {"Status": "ok", 
            "Threshold Value": threshold_value}
    
@app.post("/run_pipeline_anomalies/classic")
async def run_pipeline_classic(params : PipelineParams):
    
    dataset = params.dataset
    starting_time = f"01/01/{params.starting_time}"
    ending_time = f"31/12/{params.starting_time}"
    variable = params.variable.split('-')[0]
    variable_name = params.variable.split('-')[-1]
    min_latitude = params.latitude
    max_latitude = min_latitude + 5
    min_longitude = params.longitude
    max_longitude = min_longitude + 5
    
    anomalies_class = AnomaliesCalculation(min_latitude, min_longitude, starting_time,dataset, 95, variable)

    output = CopernicusFetcher().fetch_temperature(
                                        dataset_id=dataset, 
                                        starting_time=starting_time,
                                        ending_time=ending_time, 
                                        variable=variable,
                                        minimum_latitude=min_latitude,
                                        maximum_latitude=max_latitude,
                                        minimum_longitude=min_longitude,
                                        maximum_longitude=max_longitude)
    
    climatology_path = f"climatology_{dataset}_{min_latitude}_{min_longitude}_{variable}.nc"
    if not os.path.exists(climatology_path):
        if app.state.db_path != None:
            os.remove(app.state.db_path)
        
        app.state.db_path = climatology_path
        baseline = CopernicusFetcher().fetch_temperature(
            dataset_id=dataset,
            starting_time=starting_time,
            ending_time=ending_time,
            variable=variable,
            minimum_latitude=min_latitude,
            maximum_latitude=max_latitude,
            minimum_longitude=min_longitude,
            maximum_longitude=max_longitude,
            climatology=True
        )
        anomalies_class.ClimatologyCalculation(baseline, output, variable)
        
    climatology = xr.open_dataarray(climatology_path)
    da4d = output[variable]
    threshold_value, threshold_array = anomalies_class.ProcessAnomalies(da4d, climatology, 0)
    anomalies_class.showGraph(da4d, threshold_array, threshold_value, variable_name)
    return {"Status": "ok", 
            "Threshold Value": threshold_value}

@app.get("/show_image")
async def show_image():
    return FileResponse("results/plot.png", media_type="image/png")

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

# dataset = 'cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m'
# starting_time = '01/01/2023'
# output = CopernicusFetcher().fetch_temperature(dataset, starting_time)
# climatology = xr.open_dataarray("backend/climatology_cmems_mod_glo_phy_my_0.083deg_P1D-m_-46.0_145.0.nc")
# print(climatology)
# da4d = output['thetao']
# threshold = ProcessAnomalies(da4d, climatology, 0)
# showGraph(da4d, threshold)