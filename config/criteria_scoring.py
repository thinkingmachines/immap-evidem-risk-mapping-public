# Standard imports
import pandas as pd
import numpy as np
from pathlib import Path
import json
from uuid import uuid4

# Geospatial processing packages
from matplotlib import pyplot as plt
from shapely import wkt
import geopandas as gpd
import folium

# Util imports
from loguru import logger
from typing import List, Tuple
from collections import deque
import pandas as pd

class CriteriaScoring:
    def __init__(self, criteria_config: dict) -> None:
        self.criteria_name = criteria_config["criteria_name"]
        self.date = criteria_config["date"]
        self.scoring_thresholds_dict = criteria_config["scoring_thresholds"]
        self.binary_filter_cols = criteria_config["binary_filter"]
        self.threshold_filter_dict = criteria_config["threshold_filter"]
        self.direct_score_mapping_dict = criteria_config["direct_score_mapping"]
        self.tile_score_threshold = criteria_config["tile_score_threshold"]
        self.threshold_m_coast_tile_counting = criteria_config["threshold_m_coast_tile_counting"]
        self.group_by_tile_class = criteria_config["group_by_tile_class"]
        self.crs_proj = criteria_config["crs_proj"]
        self.land_area_ha_threshold = criteria_config["land_area_ha_threshold"]
        self.min_tiles_near_coast = criteria_config["min_tiles_near_coast"]
        self.is_weighted = criteria_config["is_weighted"]

    @staticmethod
    def _threshold_score(
        val: float, low_threshold: float, high_threshold: float, direction: str, is_weighted: bool, weight: float
    ) -> float:
        if low_threshold >= high_threshold:
            raise ValueError("low_threshold should be lower than high_threshold")

        #if lower values have higher scores
        if direction == "low_val_high_score":
          # if score is weighted
          if is_weighted == True:
            if val == low_threshold:
                score = 1.0*weight
            elif (val > low_threshold) & (val < high_threshold):
                score = (1.0 - (val - low_threshold) / (high_threshold - low_threshold))*weight
            else:
                score = 0.0
          else:
            if val == low_threshold:
                score = 1.0
            elif (val > low_threshold) & (val < high_threshold):
                score = 1.0 - (val - low_threshold) / (high_threshold - low_threshold)
            else:
                score = 0.0

        elif direction == "high_val_high_score":
          if is_weighted == True:
            if val >= high_threshold:
                score = 1.0*weight
            elif (val > low_threshold) & (val < high_threshold):
                score = (val - low_threshold) / (high_threshold - low_threshold)*weight
            else:
                score = 0.0
          else:
            if val >= high_threshold:
                score = 1.0
            elif (val > low_threshold) & (val < high_threshold):
                score = (val - low_threshold) / (high_threshold - low_threshold)
            else:
                score = 0.0
        else:
          raise ValueError("direction parameter is incorrect")
        return score

    @staticmethod
    def _check_existing_colname(df: pd.DataFrame, col_name: str) -> None:
        if col_name in df.columns:
            raise ValueError(f"{col_name} is already a column! rename this column")

    def apply_tile_criteria(
        self,
        tile_features: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:

        tile_scores = tile_features.copy()

        tile_scores = self._apply_bool_criteria(tile_scores)
        tile_scores = self._apply_threshold_criteria(tile_scores)
        tile_scores = self._apply_scoring_criteria(tile_scores)
        tile_scores = self._apply_direct_scoring_criteria(tile_scores)
        tile_scores = self._compute_avg_tile_score(tile_scores)

        return tile_scores

    def _apply_bool_criteria(self, tile_scores: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        self.bool_cols = []
        for col_name in self.binary_filter_cols:
            bool_col_name = f"{col_name}_bool"
            self._check_existing_colname(tile_scores, bool_col_name)
            self.bool_cols.append(bool_col_name)
            tile_scores[bool_col_name] = tile_scores[col_name] == True

        return tile_scores

    def _apply_threshold_criteria(
        self, tile_scores: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:

        for col_name in self.threshold_filter_dict.keys():
            threshold, direction = self.threshold_filter_dict[col_name]
            bool_col_name = f"{col_name}_bool"
            self._check_existing_colname(tile_scores, bool_col_name)
            self.bool_cols.append(bool_col_name)

            if direction == "filter_out_lower":
                tile_scores[bool_col_name] = tile_scores[col_name] >= threshold
            elif direction == "filter_out_higher":
                tile_scores[bool_col_name] = tile_scores[col_name] <= threshold
            else:
                raise ValueError(f"{direction} is an invalid direction!")

        return tile_scores

    def _apply_scoring_criteria(
        self, tile_scores: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        self.score_cols = []

        # scoring criteria
        if self.is_weighted == True:
          for col_name in self.scoring_thresholds_dict.keys():

              low_threshold, high_threshold, direction, weight  = self.scoring_thresholds_dict[col_name]
              score_col_name = f"{col_name}_score"
              self._check_existing_colname(tile_scores, score_col_name)
              # low_threshold, high_threshold = hi_low_thresholds
              self.score_cols.append(score_col_name)
              tile_scores[score_col_name] = tile_scores[col_name].apply(
                  lambda x: self._threshold_score(x, low_threshold, high_threshold, direction, self.is_weighted, weight)
              )
        else:
          for col_name in self.scoring_thresholds_dict.keys():
            low_threshold, high_threshold, direction, weight = self.scoring_thresholds_dict[col_name]
            score_col_name = f"{col_name}_score"
            self._check_existing_colname(tile_scores, score_col_name)
            # low_threshold, high_threshold = hi_low_thresholds
            self.score_cols.append(score_col_name)
            tile_scores[score_col_name] = tile_scores[col_name].apply(
                lambda x: self._threshold_score(x, low_threshold, high_threshold, direction,  self.is_weighted, weight)
            )

        return tile_scores

    def _apply_direct_scoring_criteria(
        self, tile_scores: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        for col_name, score_mapping in self.direct_score_mapping_dict.items():
            score_col_name = f"{col_name}_score"
            self._check_existing_colname(tile_scores, score_col_name)
            self.score_cols.append(score_col_name)
            tile_scores[score_col_name] = tile_scores[col_name].map(score_mapping)

        return tile_scores

    def _compute_avg_tile_score(
        self, tile_scores: gpd.GeoDataFrame, 
    ) -> gpd.GeoDataFrame:

        for col in ["tile_score", "tile_class"]:
            self._check_existing_colname(tile_scores, col)
        
        if self.is_weighted == True:
          tile_scores["tile_score"] = tile_scores.loc[:, self.score_cols].sum(axis=1)

        else:
          tile_scores["tile_score"] = tile_scores.loc[:, self.score_cols].mean(axis=1)

        tile_scores["tile_class"] = (
            tile_scores["tile_score"] >= self.tile_score_threshold
        )
        tile_scores["tile_class"] = tile_scores["tile_class"].astype(int)

        return tile_scores

    def apply_tile_filter_force_zero(
        self, tile_scores: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:

        # any boolean criteria should be met
        bool_filter_mask = (~tile_scores.loc[:, self.bool_cols]).any(axis=1)

        tile_scores.loc[bool_filter_mask, "tile_score"] = 0
        tile_scores.loc[bool_filter_mask, "tile_class"] = 0

        return tile_scores

    

    def _drop_unneeded_cols(
        self,
        tile_scores: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        if self.group_by_tile_class:
            drop_cols = self.bool_cols
        else:
            drop_cols = self.bool_cols + ["tile_class"]
        tile_scores = tile_scores.drop(columns=drop_cols)

        return tile_scores

class CriteriaViz:
    def __init__(self, criteria_config: dict) -> None:
        self.criteria_name = criteria_config["criteria_name"]
        self.date = criteria_config["date"]
        self.scoring_thresholds_dict = criteria_config["scoring_thresholds"]
        self.binary_filter_cols = criteria_config["binary_filter"]
        self.threshold_filter_dict = criteria_config["threshold_filter"]
        self.direct_score_mapping_dict = criteria_config["direct_score_mapping"]
        self.tile_score_threshold = criteria_config["tile_score_threshold"]

    def plot_threshold_scores(self, tile_scores: pd.DataFrame) -> None:
        for col, (bound1, bound2) in self.scoring_thresholds_dict.items():
            _, ax = plt.subplots(figsize=(9, 6))

            # Get number of grids that meet low/med/high criteria
            n_low = (tile_scores[col] >= bound2).sum()

            n_med = ((tile_scores[col] < bound2) & (tile_scores[col] > bound1)).sum()
            n_high = (tile_scores[col] <= bound1).sum()

            # Add colored spans for each column's thresholds
            min_val = tile_scores[col].min()
            max_val = tile_scores[col].max()

            plt.axvspan(
                min_val,
                bound1,
                color="g",
                alpha=0.2,
                label=f"Score = 100% (n={n_high:,})",
            )
            plt.axvspan(
                bound1,
                bound2,
                color="y",
                alpha=0.2,
                label=f"0% < Score < 100% (n={n_med:,})",
            )
            plt.axvspan(
                bound2, max_val, color="r", alpha=0.2, label=f"Score = 0% (n={n_low:,})"
            )

            # Plot histogram and median
            tile_scores[col].hist(bins=60, ax=ax)
            median_val = tile_scores[col].median()
            ax.axvline(
                x=median_val,
                color="red",
                linestyle="dashed",
                label=f"Median = {median_val:,.2f}",
            )
            plt.legend()

            title = f"{col}\nmedian val {median_val:,.2f}"
            ax.set_title(title)
            plt.show()

    def plot_threshold_filters(self, tile_scores: pd.DataFrame) -> None:
        for col, (bound, direction) in self.threshold_filter_dict.items():

            _, ax = plt.subplots(figsize=(9, 6))

            # Get number of tiles that meet low/med/high criteria
            if direction == "filter_out_higher":
                n_pass = (tile_scores[col] <= bound).sum()
                n_fail = (tile_scores[col] > bound).sum()
            elif direction == "filter_out_lower":
                n_pass = (tile_scores[col] >= bound).sum()
                n_fail = (tile_scores[col] < bound).sum()

            # Add colored spans for each column's thresholds
            min_val = tile_scores[col].min()
            max_val = tile_scores[col].max()

            plt.axvspan(
                min_val, bound, color="g", alpha=0.2, label=f"Pass (n={n_pass:,})"
            )
            plt.axvspan(
                bound, max_val, color="r", alpha=0.2, label=f"Fail (n={n_fail:,})"
            )

            # Plot histogram and median
            tile_scores[col].hist(bins=60, ax=ax)
            median_val = tile_scores[col].median()
            ax.axvline(
                x=median_val,
                color="red",
                linestyle="dashed",
                label=f"Median = {median_val:,.2f}",
            )
            plt.legend()

            title = f"{col}\nmedian val {median_val:,.2f}"
            ax.set_title(title)
            plt.show()