# Area
min_longitude = 55
max_longitude = 60
min_latitude = -35
max_latitude = -30

# Time
starting_time_bs = '1993-01-01'
starting_time = '2024-01-01'
ending_time = '2023-12-31'

# Depth
min_depth = 0.5057600140571594
max_depth = 0.5057600140571594

# Resolution
area_resolution = 2
depth_resolution = 15

# Periodicity
periodicity = 'monthly'
all_years = list(range(2023, 2024))

# Dataset
dataset_argo = 'argovis' # 3 possibility : erddap, gdac, argovis (only temperature and salinity are available for gdac and argovis)
dataset_baseline = "cmems_mod_glo_phy_my_0.083deg_P1M-m"
dataset = "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m"
# Variable
variable = ['TEMP', 'PSAL'] # Variables available : 'CHLA', 'TEMP', 'PSAL', 'DOXY', 'NITRATE', 'BBP700', 'DOWN_IRRADIANCE412', 'DOWN_IRRADIANCE443', 'PH_IN_SITU_TOTAL', 'CDOM'

# Percentile
percentile = 90