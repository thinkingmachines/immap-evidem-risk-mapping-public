import geopandas as gpd
import pandas as pd
import rasterio as rio
from rasterio.features import shapes
from shapely.geometry import Polygon


def convert_tif_to_polygon(raster_file, convert_label_to_int=True):
    # open the raster file
    with rio.open(raster_file) as src:
        # read the raster data as a numpy array
        raster_data = src.read(1)
        # get the metadata for the raster
        raster_meta = src.meta
        # Get CRS
        crs = src.crs

    # convert the raster data into polygons
    polygons = []
    for shape, value in shapes(raster_data, transform=raster_meta["transform"]):
        polygons.append({"geometry": shape, "label_mode": value})

    # Convert to dataframe
    df = pd.DataFrame(polygons)

    # drop NaN values
    df = df.dropna(subset=["label_mode"])

    if convert_label_to_int:
        df["label_mode"] = df["label_mode"].astype("uint8")

    # extract coordinates from the 'geometry' column and convert them into Shapely geometries
    df["geometry"] = df["geometry"].apply(lambda x: Polygon(x["coordinates"][0]))

    # Convert to geopandas dataframe
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=crs)

    # reorganize columns
    gdf = gdf[["label_mode", "geometry"]]

    return gdf
