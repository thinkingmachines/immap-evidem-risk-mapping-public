import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import Polygon


def convert_tif_to_polygon(raster_file):
    # open the raster file
    with rasterio.open(raster_file) as src:
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

    # convert the polygons to a geopandas dataframe
    gdf = gpd.GeoDataFrame(polygons, crs=crs)

    # drop NaN values
    gdf = gdf.dropna(subset=["label_mode"])

    # extract coordinates from the 'geometry' column and convert them into Shapely geometries
    gdf["geometry"] = gdf["geometry"].apply(lambda x: Polygon(x["coordinates"][0]))

    # reorganize columns
    gdf = gdf[["label_mode", "geometry"]]

    return gdf
