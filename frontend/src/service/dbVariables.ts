export const dbVariables: Record<string, string[]> = {
  'cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m': ['thetao-Temperature'],
  'cmems_mod_glo_phy_my_0.083deg_P1D-m': ['thetao-Temperature', 'so-Salinity'],
  'cmems_mod_glo_bgc_my_0.25deg_P1D-m': ['chl-Chlorophyll'],

};

export const dbYears: Record<string, string[]> = {
  'cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m': ["2022", "2023", "2024", "2025"],
  'cmems_mod_glo_phy_my_0.083deg_P1D-m': Array.from({ length: 2021 - 2010 + 1 }, (_, i) => (2010 + i).toString()),
  'cmems_mod_glo_bgc_my_0.25deg_P1D-m': Array.from({ length: 2025 - 2010 + 1 }, (_, i) => (2010 + i).toString()),
};

export const dbDescriptions: Record<string, string> = {
  'cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m': 'Global Ocean Physics Analysis and Forecast',
  'cmems_mod_glo_phy_my_0.083deg_P1D-m': 'Global Ocean Physics Reanalysis',
  'cmems_mod_glo_bgc_my_0.25deg_P1D-m': 'Global Ocean Biogeochemistry Reanalysis',
};