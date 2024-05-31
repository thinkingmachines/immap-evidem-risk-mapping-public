import numpy as np
import rasterio as rio


def convert_sentinel_to_rgb(
    sentinel_directory, rgb_directory, sentinel_filename, data_max_val=None
):
    """
    Convert sentinel images to RGB images prior to input in SAM model.
    """
    sentinel_path = f"{sentinel_directory}/{sentinel_filename}"
    output_filename = f"{rgb_directory}/RGB_positive-samples_{sentinel_filename}"

    with rio.open(sentinel_path) as src:
        # Access information about the image
        num_channels = src.count  # Get the number of bands (channels)
        # profile to match to the output image
        profile = src.profile
        profile["count"] = 3
        profile["dtype"] = "uint8"

        # Read specific channels
        # still in uint16
        band_indices = [1, 2, 3]  # Replace with the band indices you want to extract
        extracted_channels = []

        for band_index in band_indices:
            band = src.read(band_index + 1)  # Bands in rasterio are 1-based
            extracted_channels.append(band)

    # Combine the extracted channels into a single RGB image
    if len(extracted_channels) >= 3:
        rgb_image = np.stack(extracted_channels[:3], axis=-1)

        # If data_max_val is not specified
        # Get maximum of all data bands, excluding the alpha band
        if data_max_val is not None:
            data_max = data_max_val
        else:
            input_vals_max_per_band = list(np.amax(rgb_image, axis=(1, 2)))
            data_max = np.max(input_vals_max_per_band)

        # immap code
        ndata_cutoff = np.clip(
            rgb_image / data_max, 0, 0.3
        )  # divide with 10000 and cut of to range [0.0, 0.3]
        ndata_normalized = ndata_cutoff / 0.3  # stretch to [0.0, 1.0]
        rgb_image = (ndata_normalized * 255).astype(np.uint8)

        # export
        with rio.open(output_filename, "w", **profile) as dst:
            for band in range(rgb_image.shape[2]):
                dst.write(rgb_image[:, :, band], band + 1)
    else:
        print("Not enough channels to create an RGB image.")


def extract_landslide_extent(
    rgb_directory,
    output_directory,
    sam_model,  # initialized prior to this function
    box_id,  # unique ID linked to the sentinel image,
    coords,  # coordinates list x,y
):

    """
    Runs the SAM model on each pair of coordinates and their matching satellite RGB image.
    """

    image_filename = f"{rgb_directory}/RGB_positive-samples_sentinel_{box_id}.tif"
    sam_model.set_image(image_filename)

    coords_list = [coords]
    # predict landslide extent, output raster mask
    output_mask = f"{output_directory}/mask_{box_id}.tif"
    sam_model.predict(
        point_coords=coords_list,
        point_labels=1,
        point_crs="EPSG:4326",
        output=output_mask,
    )
    # convert raster mask to vector
    output_vector = f"{output_directory}/polygon_{box_id}.geojson"
    sam_model.raster_to_vector(output_mask, output=output_vector)

    return output_vector
