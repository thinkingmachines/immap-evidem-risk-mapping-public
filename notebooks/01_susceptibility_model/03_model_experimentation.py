# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: immap-evidem
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Model Experimentation
#
# This notebook runs model experiments and evaluates performance using train/test split and cross-fold validation. This is where you can experiment the impact of the type of feature you include in the model training and also test if the generated training dataset is effective in our use-case.
#
# ### Input
# - training data
# - model parameters
#
# ### Output
# - metrics for CSV
# - test data prediction

# %% [markdown]
# # Imports and Set-up
#
# *DO NOT SKIP THIS SECTION.* This section imports the packages needed to run this notebook and initializes the data file paths.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import sys

import pandas as pd
import polars as pl

# %%
sys.path.append("../../")  # include parent directory
import src.model_utils as model_utils
from src.polars_utils import log_condition, log_duplicates

# %%
from src.settings import DATA_DIR

MODEL_DIR = DATA_DIR / "models"
OUTPUT_DIR = DATA_DIR / "output/component_1"

OUTPUT_VERSION = pd.to_datetime("today").strftime("%Y%m%d")

# %%
TRAIN_VERSION = "20240504"
LATTICE_RADIUS = 3
if LATTICE_RADIUS != 0:
    TRAIN_TABLE_FPATH = (
        MODEL_DIR
        / f"training_data/training_data_w_lattice{LATTICE_RADIUS}_{TRAIN_VERSION}.parquet"
    )
else:
    TRAIN_TABLE_FPATH = (
        MODEL_DIR / f"training_data/training_data_{TRAIN_VERSION}.parquet"
    )

# %%
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

# %% [markdown]
# ## Dataset

# %%
if TRAIN_TABLE_FPATH.exists():
    features_df = pl.read_parquet(TRAIN_TABLE_FPATH)
else:
    TRAIN_TABLE_FNAME = TRAIN_TABLE_FPATH.name
    # !gsutil -m cp gs://immap-models/training_data/$TRAIN_TABLE_FNAME $TRAIN_TABLE_FPATH
    features_df = pl.read_parquet(TRAIN_TABLE_FPATH)

# %%
features_df.head(2)

# %%
features_df.shape

# %%
features_df = features_df.unique(subset=["quadkey"])

# %%
features_df.shape

# %%
features_df.select("MOV_TYPE").unique()

# %%
# Create numerical categorization of landslide

if MULTICLASS:
    LABEL_COL = "label_multiclass"
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
    LABEL_COL = "label_binary"
    # add binary labels
    features_df = features_df.with_columns(
        pl.when(pl.col("label") == "landslide")
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("label_binary")
    )

# %%
features_df[LABEL_COL].unique()

# %%
feature_cols = [
    col
    for col in features_df.columns
    if col not in KEY_COLS + EXC_COLS + AREA_COLS + LABEL_COLS
]
len(feature_cols), feature_cols

# %%
features_df.shape

# %%
features_df.select(feature_cols).null_count()

# %%
features_df.select(feature_cols).describe()

# %%
features_df["MOV_TYPE"].value_counts()

# %% [markdown]
# # Experiment with only a 80-20 train test split

# %% [markdown]
# In the following code block, we set the needed function arguments to conduct our model training and validation. These arguments will also be the same for the k-fold validation approach and will be used in the next section.

# %%
EVAL_SIMPLE_TRAIN_TEST_SPLIT = True
RANDOM_SEED = 47
MODEL_TYPE = "classification"
MODEL_NAME = "xgboost"
APPLY_LOG_TRANSFORM = False
REVERSE_LOG_TRANSFORM = False
TRAIN_WITH_PREDICTION_INTERVALS = False

label_strip = LABEL_COL.split("_")[1]
OUTPUT_PATH = (
    OUTPUT_DIR / f"{OUTPUT_VERSION}_{MODEL_TYPE}_{MODEL_NAME}_{label_strip}.parquet"
)

CV_METRICS_OUTPUT_PATH = (
    OUTPUT_DIR
    / f"{OUTPUT_VERSION}_{MODEL_TYPE}_{MODEL_NAME}_{label_strip}_cv_metrics.parquet"
)

# %%
eval_metrics = None
if EVAL_SIMPLE_TRAIN_TEST_SPLIT:
    train_df, test_df = model_utils.key_based_train_test_split(
        df=features_df,
        key_column="quadkey",
        train_proportion=0.8,
        shuffle=True,
        random_seed=RANDOM_SEED,
    )

    if MODEL_TYPE == "regression":
        plot_actual_vs_pred = True
    elif MODEL_TYPE == "classification":
        plot_actual_vs_pred = False

    kwargs = {
        "feature_cols": feature_cols,
        "label_col": LABEL_COL,  # need to update to multiclass
        "model_name": MODEL_NAME,
        "model_type": MODEL_TYPE,
        "train_df": train_df,
        "val_df": None,
        "test_df": test_df,
        "apply_log_transform": APPLY_LOG_TRANSFORM,
        "reverse_log_transform": REVERSE_LOG_TRANSFORM,
        "train_with_prediction_intervals": TRAIN_WITH_PREDICTION_INTERVALS,
        "plot_actual_vs_pred": plot_actual_vs_pred,
        "averaging_method": AVERAGING_METHOD,
        "random_seed": RANDOM_SEED,
    }
    model, eval_metrics = model_utils.train_and_eval_model(**kwargs)
eval_metrics

# %% [markdown]
# # Train and validate the model using K-fold Split approach

# %% [markdown]
# `GROUP_COL` is a parameter that sets the split of the validaiton approach. In our case below, we made use of the Department codes as the split for the k-fold validation. Make sure to choose the split that you would want to experiment and validate on.

# %%
GROUP_COL = "DPTO_CCDGO"

NUM_CV_FOLDS = features_df[GROUP_COL].n_unique()

# %%
NUM_CV_FOLDS

# %%
# %%time
kfolds = model_utils.key_based_k_fold_cross_validation(
    df=features_df, key_column="quadkey", group_column=GROUP_COL, n_splits=NUM_CV_FOLDS
)

cv_metrics = []
output_preds_df = []

for i, (train_df, val_df) in enumerate(kfolds, start=1):
    kwargs = {
        "feature_cols": feature_cols,
        "label_col": LABEL_COL,
        "model_name": MODEL_NAME,
        "model_type": MODEL_TYPE,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": None,
        "apply_log_transform": APPLY_LOG_TRANSFORM,
        "reverse_log_transform": REVERSE_LOG_TRANSFORM,
        "train_with_prediction_intervals": TRAIN_WITH_PREDICTION_INTERVALS,
        "plot_actual_vs_pred": False,
        "averaging_method": AVERAGING_METHOD,
        "random_seed": RANDOM_SEED,
        "return_val_pred": True,
    }
    model, eval_metrics, output_val_pred = model_utils.train_and_eval_model(**kwargs)
    eval_metrics["val_fold"] = i
    # eval_metrics.insert(1, "val_group", eval_metrics.pop("val_group"))
    cv_metrics.append(eval_metrics)
    output_val_pred = output_val_pred.drop(EXC_COLS)
    output_preds_df.append(output_val_pred)

predictions_kfolds_df = pl.concat(output_preds_df)

cv_metrics = pl.from_dicts(cv_metrics)
reordered_cols = cv_metrics.columns[-2:] + cv_metrics.columns[:-2]
cv_metrics = cv_metrics.select(reordered_cols)
cv_metrics.head()

# %%
cv_metrics.drop(["val_confusion_matrix", "train_confusion_matrix", "val_group"]).mean()

# %%
output_cv_metrics = cv_metrics.drop(["val_confusion_matrix", "train_confusion_matrix"])
output_cv_metrics.write_parquet(CV_METRICS_OUTPUT_PATH)

# %%
predictions_kfolds_df

# %%
predictions_kfolds_df.write_parquet(OUTPUT_PATH)
