# Area
min_longitude = -50
max_longitude = 0
min_latitude = -30
max_latitude = 0

# Time
starting_time = '2024-01-01'
ending_time = '2024-01-31'

# Depth
min_depth = 0
max_depth = 500

# Resolution
area_resolution = 2
depth_resolution = 15

# Periodicity
periodicity = 'monthly'
all_years = list(range(2023, 2024))

# Dataset
dataset = 'argovis' # 3 possibility : erddap, gdac, argovis (only temperature and salinity are available for gdac and argovis)

# Variable
variable = ['TEMP', 'PSAL'] # Variables available : 'CHLA', 'TEMP', 'PSAL', 'DOXY', 'NITRATE', 'BBP700', 'DOWN_IRRADIANCE412', 'DOWN_IRRADIANCE443', 'PH_IN_SITU_TOTAL', 'CDOM'

# Percentile
percentile = 90