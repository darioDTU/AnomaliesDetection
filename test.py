from pathlib import Path
from fastapi import FastAPI, UploadFile
import pandas as pd
import xarray as xr
from myfunctions.AnomalyAlgorithm import ClimatologyCalculation, DetectAnomalies, ProcessAnomalies
from myfunctions.coordinates_values import *
from myfunctions.AlgoPy import AlgoPy


def run_pipeline(file :Path):
    # df = pd.read_csv(file)
    # data_processing = AlgoPy()
    # data_processing.columns_index = {col: idx for idx, col in enumerate(df.columns)}
    # data_processing.fetcher = df.values
    # output = data_processing.RegionRegriding(area_resolution, depth_resolution, periodicity)
    
    output = xr.load_dataarray("output_array.nc")
    da6d, climatology = ClimatologyCalculation(output)
    threshold = ProcessAnomalies(da6d, climatology, 1)
    mask, coords = DetectAnomalies(da6d, threshold)
    coords_list = []
    for (y,t,lo,la,dep,v) in coords:
        coords_list.append({
        'year':  da6d.coords['year'].values[y],
        'month':  da6d.coords['month'].values[t],
        'lon':   da6d.coords['dim_1'].values[lo],
        'lat':   da6d.coords['dim_2'].values[la],
        'depth': da6d.coords['dim_3'].values[dep],
        'var':   da6d.coords['dim_4'].values[v]
        })
    return print(coords_list)

run_pipeline(file = Path('argo_data.csv'))