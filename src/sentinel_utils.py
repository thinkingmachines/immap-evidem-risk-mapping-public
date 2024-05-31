import ee
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


def get_ee_image(start_date, end_date, bounds_ee):
    """Get Sentinel-2 image"""
    dataset = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterDate(start_date, end_date)  # event date onwards,
        .filterBounds(bounds_ee)
        # Pre-filter to get less cloudy granules.
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .map(mask_s2_clouds)
        .mean()
        .clip(bounds_ee)
    )

    return dataset


def export_image(
    image, filename, region, folder, scale=500, to_gdrive=True, to_gcs=False, debug=True
):
    """Export Image to Google Drive.
    Args:
      image (ee.image.Image): Generated Sentinel-2 image
      filename (str): Name of image, without the file extension
      geometry (ee.geometry.Geometry): The geometry of the area of
        interest to filter to.
      folder (str): The destination folder in your Google Drive.
    Returns:
      ee.batch.Task: A task instance
    """
    if debug:
        logger.debug("Exporting to {}.tif ...".format(filename))
    if to_gdrive == True:
        task = ee.batch.Export.image.toDrive(
            image=image,
            driveFolder=folder,
            scale=10,
            region=region,
            description=filename,
            fileFormat="GeoTIFF",
            crs="EPSG:4326",
            maxPixels=900000000,
        )
        task.start()

    if to_gcs == True:
        task = ee.batch.Export.image.toCloudStorage(
            image=image,
            description=filename,
            bucket=folder,
            fileNamePrefix=f"positive-samples/{filename}",
            maxPixels=90000000000,
            scale=scale,
            region=region.geometry(),
            crs="EPSG:4326",
            fileFormat="GeoTIFF",
        )
        task.start()
    return task
