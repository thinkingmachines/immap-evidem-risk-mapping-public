import time

import ee
import geemap
from loguru import logger


def mask_s2_clouds(image):
    """Masks clouds in a Sentinel-2 image using the QA band.

    Args:
        image (ee.Image): A Sentinel-2 image.

    Returns:
        ee.Image: A cloud-masked Sentinel-2 image.
    """
    qa = image.select("QA60")

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11

    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))

    return image.updateMask(mask).divide(10000)


def get_s2_image(aoi_gdf, start_date, end_date):
    "Get the Sentinel-2 image object"

    aoi_bounds = aoi_gdf.dissolve()
    aoi_bounds_ee = geemap.geopandas_to_ee(aoi_bounds)

    dataset = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterDate(start_date, end_date)  # event date onwards,
        .filterBounds(aoi_bounds_ee)
        # Pre-filter to get less cloudy granules.
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .map(mask_s2_clouds)
        .mean()
        .clip(aoi_bounds_ee)
    )

    return dataset


def export_image_to_gcs(
    image,
    filename,
    aoi_gdf,
    bucket,
    scale=500,
    debug=True,
    bucket_subdirectory="export",
    return_async=True,
    recheck_time=30,
):
    """Export Image to Google Cloud Storage bucket.
    Args:
      image (ee.image.Image): Generated Sentinel-2 image
      filename (str): Name of image, without the file extension
      aoi_gdf (gpd.GeoDataFrame): The geometry of the area of
        interest to filter to, given as a geodataframe
      bucket (str): The destination GCS bucket
      bucket_subdirectory (str): Destination subdirectory within GCS bucket
      scale (int): resolution of output geoTIFF in meters
      return_async (bool): If true, wait for task to complete before returning
    Returns:
      ee.batch.Task: A task instance
    """
    aoi_bounds = aoi_gdf.dissolve()
    aoi_bounds_ee = geemap.geopandas_to_ee(aoi_bounds)

    if debug:
        logger.debug("Exporting to {}.tif ...".format(filename))

    task = ee.batch.Export.image.toCloudStorage(
        image=image,
        description=filename,
        bucket=bucket,
        fileNamePrefix=f"{bucket_subdirectory}/{filename}",
        maxPixels=90000000000,
        scale=scale,
        region=aoi_bounds_ee.geometry(),
        crs="EPSG:4326",
        fileFormat="GeoTIFF",
    )
    task.start()

    # Wait for task to finish
    if not return_async:
        logger.info("Waiting for task to finish")
        while task.status()["state"] != "COMPLETED":
            time.sleep(recheck_time)

    return task
