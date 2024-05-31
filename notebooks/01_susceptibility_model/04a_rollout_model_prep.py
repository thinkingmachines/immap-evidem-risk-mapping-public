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
# # Roll-out Model Prep
#
# This notebook trains a model using all the training data using the best configuration on the latest experiment. At the end of this notebook, it outputs the model into a pkl file that will be used for predicting in our target municipalities.
#
# ### Input
# - Model configuration from experiments (list of features to exclude, etc.)
# - Training data (same version as what was used in the last experiment)
#
# ### Output
# - Model pkl file for roll-out

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Imports and Set-up

# %%
import os
import sys

import pandas as pd
import polars as pl
from loguru import logger
from pytz import timezone
from shapash import SmartExplainer
from skops.io import dump as dump_model

# %%
sys.path.append("../../")  # include parent directory
import src.model_utils as model_utils
from src.settings import DATA_DIR

# %%
pl.Config(fmt_str_lengths=100)

# %%
MODEL_DIR = DATA_DIR / "models"
OUTPUT_VERSION = pd.to_datetime("today").strftime("%Y%m%d")

TRAIN_VERSION = "20240504"
LATTICE_RADIUS = 3
TRAIN_TABLE_FNAME = f"training_data_w_lattice{LATTICE_RADIUS}_{TRAIN_VERSION}.parquet"
TRAIN_TABLE_FPATH = MODEL_DIR / f"training_data/{TRAIN_TABLE_FNAME}"

# %% [markdown]
# ## Model Settings

# %%
# Label settings
MULTICLASS = True
LABEL_COL = "label_multiclass"

# Model Settings
RANDOM_SEED = 47
MODEL_TYPE = "classification"
MODEL_NAME = "xgboost"
APPLY_LOG_TRANSFORM = False
REVERSE_LOG_TRANSFORM = False
TRAIN_WITH_PREDICTION_INTERVALS = False
SAVE_TRAINED_MODEL = True
label_strip = LABEL_COL.split("_")[1]
OUTPUT_PATH = (
    MODEL_DIR / f"pkl/{OUTPUT_VERSION}_{MODEL_TYPE}_{MODEL_NAME}_{label_strip}.parquet"
)

# %% [markdown]
# ## Load Data

# %%
features_df = pl.read_parquet(TRAIN_TABLE_FPATH)

# %% [markdown]
# # Model Dev

# %% [markdown]
# ## Train data prep

# %%
# Define column categorization, this indicates which columns should be excluded from the model training
KEY_COLS = ["quadkey"]
LABEL_COLS = ["label", "label_binary", "MOV_TYPE", "label_multiclass"]
MULTICLASS = True
if MULTICLASS:
    LABEL_COL = "label_multiclass"  # choose either label_binary or label_multiclass
    AVERAGING_METHOD = "weighted"  # binary if 2 classes, use weighted if multiclass
else:
    LABEL_COL = "label_binary"  # choose either label_binary or label_multiclass
    AVERAGING_METHOD = "binary"  # binary if 2 classes, use weighted if multiclass
AREA_COLS = [
    "MPIO_CCNCT",
    "MPIO_CNMBR",
    "MPIO_CNMBR_EN",
    "DPTO_CNMBR",
    "DPTO_CNMBR_EN",
    "Municipio",
    "Municipio_EN",
    "DPTO_CCDGO",
    "MPIO_CCDGO",
    "MPIO_CRSLC",
    "MPIO_NAREA",
    "MPIO_NANO",
    "SHAPE_AREA",
    "SHAPE_LEN",
    "OBJECTID",
    "source",
]
EXC_COLS = [
    "source",
    "x",
    "y",
    "z",
    "slope_count",
    "slope_min",
    "slope_max",
    "aspect_count",
    "aspect_max",
    "aspect_min",
    "elevation_min",
    "elevation_max",
    "elevation_count",
    "rainfall_mm_count",
    "rainfall_mm_min",
    "rainfall_mm_max",
    "hillshade_count",
    "hillshade_min",
    "hillshade_max",
    "hillshade_count",
    "ndvi2023_min",
    "ndvi2023_max",
    "ndvi2023_count",
    "ndvi2023_median",
    "__index_level_0__",
    "geometry",
]

# %%
# Create numerical categorization of landslide

if MULTICLASS:
    # add multiclass labels
    features_df = features_df.with_columns(
        pl.when(pl.col("MOV_TYPE") == "landslide")
        .then(1)
        .when(pl.col("MOV_TYPE") == "flows")
        .then(2)
        .otherwise(0)
        .alias("label_multiclass")
    )
else:
    # add binary labels
    features_df = features_df.with_columns(
        pl.when(pl.col("label") == "landslide")
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("label_binary")
    )

# %%
feature_cols = [
    col
    for col in features_df.columns
    if col not in KEY_COLS + EXC_COLS + AREA_COLS + LABEL_COLS
]

len(feature_cols), feature_cols

# %%
X_train = features_df.select(feature_cols)
y_train = features_df.select(LABEL_COL).to_series()

X_test = features_df.select(feature_cols)
y_test = features_df.select(LABEL_COL).to_series()
# rent_sqm_test = features_df.filter(filter_expr).select(RENT_SALES_COLS)

X_train.shape, y_train.shape, X_test.shape, y_test.shape

# %%
X_train

# %% [markdown]
# ## Model Training

# %%
# %%time
model = model_utils.get_tabular_model(
    model_name=MODEL_NAME,
    model_type=MODEL_TYPE,
    random_seed=RANDOM_SEED,
    train_with_prediction_intervals=TRAIN_WITH_PREDICTION_INTERVALS,
    # model_kwargs={"monotonic_cst":monotonic_cst}
)
model.fit(X_train.to_pandas(), y_train.to_pandas())
model

# %% [markdown]
# ## SHAP (feature importance)

# %%
xpl = SmartExplainer(model=model)

# %%
X_test = X_test.to_pandas()

# %%
xpl.compile(x=X_test)

# %%
xpl.plot.features_importance()

# %% [markdown]
# ## Export model to disk

# %%
# %%time
if SAVE_TRAINED_MODEL:
    model_settings = {
        "features": feature_cols,
        "apply_log_transform": APPLY_LOG_TRANSFORM,
        "reverse_log_transform": REVERSE_LOG_TRANSFORM,
        "label_column": LABEL_COL,
    }
    model.model_settings_ = model_settings
    logger.info(f"Saving model to {OUTPUT_PATH.name}")
    if os.path.exists(OUTPUT_PATH):
        logger.info(f"Existing file found. Will overwrite!")
    dump_model(model, OUTPUT_PATH)
