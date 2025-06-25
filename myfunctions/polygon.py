import geopandas as gpd
import numpy as np
import os



### POLYGONS ###
def get_lat_lon_from_polygon(polygon):
        ### to get the boundaries of the polygon (for rectangular polygon)
    polygon_boundary = np.array([*list(polygon.iloc[0]['geometry'].exterior.coords)])
    start_lat = polygon_boundary[:,1].min()
    end_lat = polygon_boundary[:,1].max()
    start_lon = polygon_boundary[:,0].min()
    end_lon = polygon_boundary[:,0].max()

    return start_lat, end_lat, start_lon, end_lon

def get_area_coordinates(polygon_name):
    polygon_path = f'Polygons/{polygon_name}/{polygon_name}.geojson'
    polygon = gpd.read_file(polygon_path)
    return get_lat_lon_from_polygon(polygon)


from shapely.geometry import mapping, Polygon
import json

def create_geojson(polygon_name, space_constraints):
    
    # Ensure the polygon directory exists
    os.makedirs(f'Polygons/{polygon_name}', exist_ok=True)
    geojson_path = f'Polygons/{polygon_name}/{polygon_name}.geojson'

    if not os.path.exists(geojson_path):

        # Create GeoJSON structure
        geojson_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": mapping(Polygon([
                        (space_constraints[2], space_constraints[0]),  # MinLon, MinLat
                        (space_constraints[3], space_constraints[0]),  # MaxLon, MinLat
                        (space_constraints[3], space_constraints[1]),  # MaxLon, MaxLat
                        (space_constraints[2], space_constraints[1]),  # MinLon, MaxLat
                        (space_constraints[2], space_constraints[0])   # Closing point
                    ])),
                    "properties": {"name": polygon_name}
                }
            ]
        }

        with open(geojson_path, "w") as geojson_file:
            json.dump(geojson_data, geojson_file, indent=4)
        
        print(f"GeoJSON created: {geojson_path}")

