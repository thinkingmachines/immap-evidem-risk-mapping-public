import geopandas as gpd
import numpy as np
from geowrangler.validation import GeometryValidation
from sklearn.metrics import roc_auc_score

from src.settings import LOCAL_CRS


def calculate_single_iou(groundtruth_gdf, pred_gdf):
    """
    Check how much of SAM polygons overlapped to the
    predictions.
    """
    groundtruth_gdf = groundtruth_gdf.reset_index()
    pred_gdf = pred_gdf.reset_index()

    groundtruth_gdf = GeometryValidation(
        groundtruth_gdf, add_validation_columns=False
    ).validate_all()
    pred_gdf = GeometryValidation(pred_gdf, add_validation_columns=False).validate_all()

    groundtruth_gdf = groundtruth_gdf.to_crs(LOCAL_CRS)
    pred_gdf = pred_gdf.to_crs(LOCAL_CRS)

    intersection_area = gpd.overlay(
        groundtruth_gdf, pred_gdf, how="intersection"
    ).unary_union.area
    gt_area = groundtruth_gdf.unary_union.area

    union_area = groundtruth_gdf.union(pred_gdf).unary_union.area

    iou_total_union = intersection_area / union_area
    iou_gt_only = intersection_area / gt_area

    # pred_gdf['iou_val'] = iou_value
    print(f"IoU using overall union: {iou_total_union:.5f}")
    print(f"IoU using SAM polygons area: {iou_gt_only:.5f}")


def calculate_poly_iou(groundtruth_gdf, pred_gdf):
    """
    Calculate IoU between ground truth landslide polygons and predicted susceptibility grids.
    """
    # Convert ground truth polygons to the same CRS as the predicted grid
    groundtruth_gdf = groundtruth_gdf.to_crs(LOCAL_CRS)
    pred_gdf = pred_gdf.to_crs(LOCAL_CRS)

    # Calculate intersection area between each ground truth polygon and the predicted grid cells
    intersection_areas = gpd.overlay(groundtruth_gdf, pred_gdf, how="intersection").area

    # Calculate area of each ground truth polygon
    groundtruth_areas = groundtruth_gdf.area

    # Calculate IoU for each ground truth polygon
    iou_values = intersection_areas / groundtruth_areas

    # Add IoU values to the ground truth GeoDataFrame
    groundtruth_gdf["iou_val"] = iou_values

    return groundtruth_gdf


def calculate_auc_roc(ground_truth_gdf, predicted_labels_gdf, pred_col="pred_proba_1"):
    """
    Calculate AUC-ROC for two GeoDataFrames.

    Args:
    ground_truth_gdf (GeoDataFrame): GeoDataFrame containing ground truth coordinates.
    predicted_labels_gdf (GeoDataFrame): GeoDataFrame containing predicted labels assigned to grid cells.

    Returns:
    auc_roc_score (float): AUC-ROC score.
    """
    # Merge predicted labels with ground truth based on spatial relationship
    merged_gdf = gpd.sjoin(
        predicted_labels_gdf, ground_truth_gdf, how="left", op="intersects"
    )

    # Extract true labels (1 if intersection exists, 0 otherwise)
    true_labels = np.zeros(len(merged_gdf))
    true_labels[~merged_gdf.index_right.isna()] = 1

    # Predicted probabilities for the positive class (assuming the label column is 'predicted_label')
    predicted_probabilities = merged_gdf[pred_col]

    # Calculate AUC-ROC
    auc_roc_score = roc_auc_score(
        true_labels, predicted_probabilities, multi_class="ovr"
    )

    return auc_roc_score
