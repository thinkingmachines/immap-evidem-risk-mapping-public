# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Model Roll-out
#
# This notebook uses the most recent model to predict susceptibility for the target municipalities
#
# ### Input
# - Features for target municipality
# - Model instance
#
# ### Output
# - predicted landslide susceptibility for each quadkey in target municipalities

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Imports and Set Up

# %%
import sys

import geopandas as gpd
import pandas as pd
import polars as pl
from shapely import wkt
from skops.io import load as load_model

# %%
sys.path.append("../../")  # include parent directory

from src.settings import DATA_DIR

# %%
MODEL_DIR = DATA_DIR / "models"

FEATURES_VERSION = "20240504"
LATTICE_RADIUS = 3
FEATURES_FPATH = (
    MODEL_DIR
    / f"rollout_data/rollout_data_w_lattice{LATTICE_RADIUS}_{FEATURES_VERSION}.parquet"
)


ADM_GRIDS = DATA_DIR / "admin_bounds/grids_target_muni_wadm_zoomlevel18_20240304.gpkg"

MODEL_VERSION = "20240507"
MODEL_TYPE = "classification"
MODEL_NAME = "xgboost"
LABEL = "multiclass"
MODEL_FPATH = (
    MODEL_DIR / f"pkl/{MODEL_VERSION}_{MODEL_TYPE}_{MODEL_NAME}_{LABEL}.parquet"
)

OUTPUT_VERSION = pd.to_datetime("today").strftime("%Y%m%d")
OUTPUT_FPATH = DATA_DIR / f"output/component_1/{OUTPUT_VERSION}_rollout_preds.gpkg"

# %% [markdown]
# ## Load Data

# %%
features_df = pl.read_parquet(FEATURES_FPATH)

# %%
features_df.head(), features_df.shape

# %%
adm_grids = gpd.read_file(ADM_GRIDS)
adm_grids["geometry"] = adm_grids.geometry.apply(lambda x: wkt.dumps(x))
adm_grids = pl.from_pandas(adm_grids)

# %%
adm_grids.head()

# %% [markdown]
# ## Load Model

# %%
model = load_model(MODEL_FPATH, trusted=True)
model

# %%
model_settings = model.model_settings_
model_settings

# %% [markdown]
# # Model Roll-out

# %%
# the target coverage for the prediction interval is 1 - alpha
PRED_INTERVAL_ALPHA = 0.1

# %%
# Set up for model inference
X_rollout = features_df.select(model_settings["features"])
X_rollout.head()

# %%
# %%time
pred = pl.Series("y_val_pred", model.predict(X_rollout.to_pandas()))
pred_proba = pl.Series("y_val_pred", model.predict_proba(X_rollout.to_pandas()))

# %%
label_col = model_settings["label_column"]

rollout_df = features_df.with_columns(
    pl.Series(name=f"{label_col}_pred_class", values=pred)
)
rollout_df = rollout_df.with_columns(
    pl.Series(name=f"{label_col}_pred_proba", values=pred_proba)
)
rollout_df = rollout_df.with_columns(
    pl.col(f"{label_col}_pred_proba").list.to_struct()
).unnest(f"{label_col}_pred_proba")

# Generating rename mapping based on the number of classes
proba_length = len(pred_proba[0])
rename_mapping = {f"field_{i}": f"pred_proba_{i}" for i in range(proba_length)}

# Renaming the columns
rollout_df = rollout_df.rename(rename_mapping)

# %%
rollout_df.head()

# %%
# add  geometry
rollout_df = rollout_df.join(adm_grids.select(["quadkey", "geometry"]), on="quadkey")

# %%
rollout_df = rollout_df.to_pandas()

# %%
# convert to gdf
rollout_gdf = gpd.GeoDataFrame(
    rollout_df, geometry=rollout_df.geometry.apply(wkt.loads), crs="EPSG:4326"
)

# %%
if OUTPUT_FPATH.exists():
    # !rm -rf $OUTPUT_FPATH
    rollout_gdf.to_file(OUTPUT_FPATH, driver="GPKG")
else:
    rollout_gdf.to_file(OUTPUT_FPATH, driver="GPKG")

# %%
rollout_gdf.to_csv(DATA_DIR / f"output/component_1/{OUTPUT_VERSION}_rollout_preds.csv")
