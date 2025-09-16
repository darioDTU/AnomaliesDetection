import numpy as np
import xarray as xr
from myfunctions.coordinates_values import *

def AnomalyDetection(input_array):
    # pick your time‐axis length & name
    if periodicity == 'monthly':
        n_t = 12
        tdim = 'month'
    else:
        n_t = 52
        tdim = 'week'

    # reshape into (year, month/week, lon, lat, depth, var)
    arr = input_array.values.reshape(len(all_years), n_t, *input_array.shape[1:])
    da6d = xr.DataArray(
        arr,
        dims=('year', tdim) + input_array.dims[1:],
        coords={
            'year': all_years,
            tdim: np.arange(1, n_t+1),
            **{d: input_array.coords[d] for d in input_array.dims[1:]}
        }
    )

    # compute the mean across the 'year' dimension
    climatology = da6d.mean(dim='year', skipna=True)
    climatology = climatology.expand_dims('year')
    climatology = climatology.assign_coords(year=['climatology'])

    return da6d, climatology