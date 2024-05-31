import csv
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import fasttreeshap
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from catboost import CatBoostClassifier, CatBoostRegressor
from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)
from lightgbm import LGBMClassifier, LGBMRegressor
from loguru import logger
from mapie.regression import MapieRegressor
from mapie.subsample import Subsample
from matplotlib.ticker import FuncFormatter
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GroupKFold, KFold, LeaveOneOut, train_test_split
from xgboost import XGBClassifier, XGBRegressor

from .polars_utils import log_condition

ALLOWED_TABULAR_MODELS = [
    # "linear",
    "ebm",
    "rf",
    "lightgbm",
    "hgbdt",
    "catboost",
    "xgboost",
]

# comment: replace with clf metrics
PRIORITRY_EVAL_METRICS_COLS = [
    "train_accuracy",
    "val_accuracy",
    "train_f1",
    "val_f1",
    "train_precision",
    "train_recall",
    "val_precision",
    "val_recall",
]


def get_tabular_model(
    model_name: str,
    model_type: str = "regression",
    random_seed: Optional[int] = None,
    train_with_prediction_intervals: bool = False,
    model_kwargs: Optional[Dict] = None,
) -> Any:

    assert model_type in ["regression", "classification"]

    if model_kwargs is None:
        model_kwargs = {}

    if model_name == "catboost":
        if model_type == "regression":
            model = CatBoostRegressor(
                random_state=random_seed, silent=True, **model_kwargs
            )
        elif model_type == "classification":
            model = CatBoostClassifier(
                random_state=random_seed, silent=True, **model_kwargs
            )
    elif model_name == "ebm":
        if model_type == "regression":
            model = ExplainableBoostingRegressor(
                random_state=random_seed, **model_kwargs
            )
        elif model_type == "classification":
            model = ExplainableBoostingClassifier(
                random_state=random_seed, **model_kwargs
            )

    elif model_name == "hgbdt":
        if model_type == "regression":
            model = HistGradientBoostingRegressor(
                random_state=random_seed, **model_kwargs
            )
        elif model_type == "classification":
            model = HistGradientBoostingClassifier(
                random_state=random_seed, **model_kwargs
            )

    elif model_name == "lightgbm":
        if model_type == "regression":
            model = LGBMRegressor(
                random_state=random_seed, verbosity=-1, **model_kwargs
            )
        elif model_type == "classification":
            model = LGBMClassifier(
                random_state=random_seed, verbosity=-1, **model_kwargs
            )

    elif model_name == "rf":
        if model_type == "regression":
            model = RandomForestRegressor(
                n_estimators=200,
                random_state=random_seed,
                oob_score=True,
                **model_kwargs,
            )
        elif model_type == "classification":
            model = RandomForestClassifier(
                n_estimators=200,
                random_state=random_seed,
                oob_score=True,
                **model_kwargs,
            )
    elif model_name == "xgboost":
        if model_type == "regression":
            model = XGBRegressor(random_state=random_seed, **model_kwargs)
        elif model_type == "classification":
            model = XGBClassifier(random_state=random_seed, **model_kwargs)
    elif model_name == "linear":
        if model_type == "regression":
            model = LinearRegression()
        elif model_type == "classification":
            model = LogisticRegression()
    else:
        raise NotImplemented(
            f"{model_name} not supported. Allowed tabular models are {ALLOWED_TABULAR_MODELS}"
        )

    if train_with_prediction_intervals:
        if model_type == "regression":
            # the jackknife plus/minus after-bootstrap method is most recommended for prediction intervals
            cv = Subsample(n_resamplings=50, random_state=random_seed)
            model = MapieRegressor(
                model, method="plus", cv=cv, random_state=random_seed
            )
        elif model_type == "classification":
            raise NotImplemented(
                f"Prediction intervals not supported for classification models!"
            )

    return model


def merge_and_prefix_metrics(prefix: str, ml_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Merge ML metrics with a prefix."""
    merged_metrics = {f"{prefix}_{k}": v for k, v in {**ml_metrics}.items()}
    return merged_metrics


def compile_eval_metrics(
    train_ml_metrics: Optional[Dict[str, Any]] = None,
    val_ml_metrics: Optional[Dict[str, Any]] = None,
    test_ml_metrics: Optional[Dict[str, Any]] = None,
    val_ml_group: Optional[str] = None,
) -> Dict[str, Any]:
    # Use dict.get() to provide default empty dicts if None
    train_metrics = merge_and_prefix_metrics("train", train_ml_metrics or {})
    val_metrics = merge_and_prefix_metrics("val", val_ml_metrics or {})
    test_metrics = merge_and_prefix_metrics("test", test_ml_metrics or {})

    eval_metrics = {**train_metrics, **val_metrics, **test_metrics}
    return eval_metrics


def log_model_eval_metrics(
    log_filepath: Union[str, Path],
    model_run_settings: Dict[str, Any],
    eval_metrics: Dict[str, Any],
) -> None:

    model_results_log = model_run_settings | eval_metrics

    write_headers = not os.path.exists(log_filepath)
    logger.info(f"Logging results to {log_filepath.name}")
    with open(log_filepath, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=model_results_log.keys())

        if write_headers:
            writer.writeheader()

        writer.writerow(model_results_log)


def format_large_numbers(x, pos):
    if np.abs(x) >= 1e6:
        return f"{x * 1e-6:.0f}M"
    elif np.abs(x) >= 1e3:
        return f"{x * 1e-3:.0f}K"
    else:
        return f"{x:.2f}"


def plot_regression_actual_vs_pred(
    y_true: pl.Series,
    y_pred: pl.Series,
    title: Optional[str] = None,
) -> None:
    # Determine the limits for the diagonal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.5)
    # Plot the diagonal line
    ax.plot([min_val, max_val], [min_val, max_val], "--", color="gray", alpha=0.5)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(FuncFormatter(format_large_numbers))
    ax.xaxis.set_major_formatter(FuncFormatter(format_large_numbers))
    plt.show()


def _check_key_column(
    df: pl.DataFrame,
    key_column: str,
) -> None:

    keys = df[key_column].to_list()
    none_keys = [k for k in keys if k is None]
    if none_keys:
        raise ValueError(f"Found {len(none_keys)} None values in {key_column}")
    if len(set(keys)) != len(df):
        raise ValueError(f"{key_column} is not unique in df! It should be unique")


def key_based_train_test_split(
    df: pl.DataFrame,
    key_column: str,
    train_proportion: float = 0.8,
    shuffle: bool = True,
    random_seed: Optional[int] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame]:

    keys = df[key_column].to_list()
    _check_key_column(df, key_column)

    train_keys, test_keys = train_test_split(
        keys, train_size=train_proportion, random_state=random_seed, shuffle=shuffle
    )

    train_df = df.filter(pl.col(key_column).is_in(train_keys))
    test_df = df.filter(pl.col(key_column).is_in(test_keys))

    return train_df, test_df


def key_based_k_fold_cross_validation(
    df: pl.DataFrame,
    key_column: str,
    group_column: Optional[str] = None,
    n_splits: int = 5,
    shuffle: bool = True,
    random_seed: Optional[int] = None,
) -> Iterator[Tuple[pl.DataFrame, pl.DataFrame]]:

    keys = df[key_column].to_list()
    _check_key_column(df, key_column)

    if group_column is None:
        groups = None
    else:
        groups = df[group_column].to_list()

    if n_splits > len(df):
        raise ValueError(
            f"k ({n_splits}) is larger than the number of rows in df ({len(df)}). It should be smaller"
        )

    if group_column is None:
        if n_splits == -1:
            kf = LeaveOneOut()
        else:
            kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_seed)
    else:
        if n_splits == -1:
            raise NotImplementedError(
                "Leave-one-out cross-validation isn't applicable with a group column"
            )
        elif random_seed is not None:
            raise NotImplementedError("Random seed not supported for GroupKFold")
        else:
            kf = GroupKFold(n_splits=n_splits)

    test_group_labels = []

    for train_index, val_index in kf.split(keys, groups=groups):
        train_keys = [keys[i] for i in train_index]
        val_keys = [keys[i] for i in val_index]

        # add label
        val_group_label = groups[val_index[0]]
        department_name = (
            df.filter(pl.col(group_column) == val_group_label)
            .select("DPTO_CNMBR_EN")
            .item(0, 0)
        )
        test_group_labels.append(val_group_label)

        val_df = df.filter(pl.col(key_column).is_in(val_keys))
        val_df = val_df.with_columns(pl.lit(val_group_label).alias("val_group"))
        val_df = val_df.with_columns(pl.lit(department_name).alias("department_name"))
        train_df = df.filter(pl.col(key_column).is_in(train_keys))

        yield train_df, val_df


def evaluate_tabular_model(
    y_true: pl.Series,
    y_pred: pl.Series,
    model_type: str = "regression",
    averaging_method: str = None,
) -> Dict[str, float]:

    if model_type == "regression":
        r2 = r2_score(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        mean_ae = mean_absolute_error(y_true, y_pred)
        median_ae = median_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)

        eval_metrics = {
            "r2": r2,
            "rmse": rmse,
            "median_ae": median_ae,
            "mean_ae": mean_ae,
            "mape": mape,
        }
    elif model_type == "classification":
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=averaging_method)
        recall = recall_score(y_true, y_pred, average=averaging_method)
        f1 = f1_score(y_true, y_pred, average=averaging_method)
        conf_matrix = confusion_matrix(y_true, y_pred)

        eval_metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": conf_matrix,
        }
    else:
        raise NotImplementedError(f"Not supporting model_type {model_type}")

    return eval_metrics


def train_and_eval_model(
    feature_cols: List[str],
    label_col: str,
    model_name: str,
    model_type: str,
    train_df: pl.DataFrame,
    val_df: Optional[pl.DataFrame] = None,
    test_df: Optional[pl.DataFrame] = None,
    apply_log_transform: bool = False,
    reverse_log_transform: bool = False,
    train_with_prediction_intervals: bool = False,
    plot_actual_vs_pred: bool = False,
    eval_metrics_type: str = "aggregated",
    averaging_method: str = None,
    include_shap: bool = False,
    random_seed: Optional[int] = None,
    model_kwargs: Optional[Dict] = None,
    return_val_pred: bool = False,
    calibrate_probability: bool = False,
) -> Tuple[Any, Any]:

    assert eval_metrics_type in ["aggregated", "unaggregated"]
    if model_type == "classification":
        if apply_log_transform or reverse_log_transform:
            raise NotImplementedError(
                "Log transform not applicable for classification models"
            )

    X_train = train_df.select(feature_cols)
    y_train = train_df.select(label_col).to_series()

    if val_df is not None:
        X_val = val_df.select(feature_cols)
        y_val = val_df.select(label_col).to_series()
        y_val_group = val_df.select("department_name").item(0, 0)
        # num_classes = val_df.select(label_col).n_unique()

    if test_df is not None:
        X_test = test_df.select(feature_cols)
        y_test = test_df.select(label_col).to_series()

    if apply_log_transform:
        y_train = y_train.log()
        if val_df is not None:
            y_val = y_val.log()
        if test_df is not None:
            y_test = y_test.log()

    model = get_tabular_model(
        model_name=model_name,
        model_type=model_type,
        random_seed=random_seed,
        train_with_prediction_intervals=train_with_prediction_intervals,
        model_kwargs=model_kwargs,
    )

    if calibrate_probability:
        calibrated_clf = CalibratedClassifierCV(model, method="isotonic")
        calibrated_clf.fit(X_train.to_pandas(), y_train.to_pandas())
    else:
        model.fit(X_train.to_pandas(), y_train.to_pandas())

    if val_df is not None:
        if calibrate_probability:
            y_val_pred = pl.Series(
                "y_val_pred", calibrated_clf.predict(X_val.to_pandas())
            )
            y_val_pred_proba = pl.Series(
                "y_val_pred", calibrated_clf.predict_proba(X_val.to_pandas())
            )
        else:
            y_val_pred = pl.Series("y_val_pred", model.predict(X_val.to_pandas()))
            y_val_pred_proba = pl.Series(
                "y_val_pred", model.predict_proba(X_val.to_pandas())
            )
    if test_df is not None:
        y_test_pred = pl.Series("y_test_pred", model.predict(X_test.to_pandas()))

    if calibrate_probability:
        y_train_pred = pl.Series(
            "y_train_pred", calibrated_clf.predict(X_train.to_pandas())
        )
    else:
        y_train_pred = pl.Series("y_train_pred", model.predict(X_train.to_pandas()))

    if reverse_log_transform:
        if val_df is not None:
            y_val = y_val.exp()
            y_val_pred = y_val_pred.exp()

        if test_df is not None:
            y_test = y_test.exp()
            y_test_pred = y_test_pred.exp()

        y_train = y_train.exp()
        y_train_pred = y_train_pred.exp()

    if plot_actual_vs_pred:
        if model_type == "classification":
            raise NotImplementedError(
                "Predicted vs actual plot not supported for classification"
            )

        title = "Predicted vs Actual on Train Set"
        plot_regression_actual_vs_pred(y_train, y_train_pred, title=title)

        if val_df is not None:
            title = "Predicted vs Actual on Validation Set"
            plot_regression_actual_vs_pred(y_val, y_val_pred, title=title)

        if test_df is not None:
            title = "Predicted vs Actual on Test Set"
            plot_regression_actual_vs_pred(y_test, y_test_pred, title=title)

    if eval_metrics_type == "aggregated":
        train_ml_metrics = evaluate_tabular_model(
            y_train,
            y_train_pred,
            model_type=model_type,
            averaging_method=averaging_method,
        )

        val_ml_metrics = None
        test_ml_metrics = None

        if val_df is not None:
            val_ml_metrics = evaluate_tabular_model(
                y_val,
                y_val_pred,
                model_type=model_type,
                averaging_method=averaging_method,
            )

        if test_df is not None:
            test_ml_metrics = evaluate_tabular_model(
                y_test,
                y_test_pred,
                model_type=model_type,
                averaging_method=averaging_method,
            )

        eval_metrics = compile_eval_metrics(
            train_ml_metrics=train_ml_metrics,
            val_ml_metrics=val_ml_metrics,
            test_ml_metrics=test_ml_metrics,
        )

        if val_df is not None:
            eval_metrics["val_group"] = y_val_group

        if model_name == "rf":
            if train_with_prediction_intervals:
                eval_metrics[
                    "rf_oob_score"
                ] = model.estimator_.single_estimator_.oob_score_
            else:
                eval_metrics["rf_oob_score"] = model.oob_score_

    elif eval_metrics_type == "unaggregated":

        val_metrics = None
        test_metrics = None
        val_shap_values = None
        test_shap_values = None
        if include_shap:
            explainer = fasttreeshap.TreeExplainer(model, algorithm="auto", n_jobs=-1)

        if val_df is not None:
            if include_shap:
                shap = explainer(X_val.to_pandas(), y_val.to_pandas())

                shap_values = pl.DataFrame(
                    shap.values,
                    schema={f"shap_{col}": pl.Float64 for col in X_val.columns},
                )
                base_values = pl.DataFrame(
                    shap.base_values, schema={"shap_base_value": pl.Float64}
                )
                val_shap_values = pl.concat(
                    [shap_values, base_values], how="horizontal"
                )

        if test_df is not None:
            if include_shap:
                shap = explainer(X_test.to_pandas(), y_test.to_pandas())

                shap_values = pl.DataFrame(
                    shap.values,
                    schema={f"shap_{col}": pl.Float64 for col in X_test.columns},
                )
                base_values = pl.DataFrame(
                    shap.base_values, schema={"shap_base_value": pl.Float64}
                )
                test_shap_values = pl.concat(
                    [shap_values, base_values], how="horizontal"
                )

        eval_metrics = {
            # "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "val_shap": val_shap_values,
            "test_shap": test_shap_values,
        }

    if return_val_pred:
        if val_df is not None:
            val_df = val_df.with_columns(
                pl.Series(name=f"{label_col}_pred_class", values=y_val_pred)
            )
            val_df = val_df.with_columns(
                pl.Series(name=f"{label_col}_pred_proba", values=y_val_pred_proba)
            )
            val_df = val_df.with_columns(
                pl.col(f"{label_col}_pred_proba").list.to_struct()
            ).unnest(f"{label_col}_pred_proba")

            # Generating rename mapping based on the number of classes
            proba_length = len(y_val_pred_proba[0])
            rename_mapping = {
                f"field_{i}": f"pred_proba_{i}" for i in range(proba_length)
            }

            # Renaming the columns
            val_df = val_df.rename(rename_mapping)

        return model, eval_metrics, val_df
    else:
        return model, eval_metrics


def get_feat_impt_df(x_cols: List[str], feat_impt: np.array) -> pl.DataFrame:
    feat_impt = zip(x_cols, feat_impt)
    feat_impt = [{"feature": f, "feature_importance": fi} for f, fi in feat_impt]

    feat_impt_df = pl.from_dicts(feat_impt)
    feat_impt_df = feat_impt_df.sort(by="feature_importance", descending=True)

    return feat_impt_df


def predict_with_intervals(
    model: Any,
    X_predict: pl.DataFrame,
    pred_col: Optional[str] = None,
    alpha: float = 0.1,
    reverse_log_transform: bool = False,
    include_interval_stats_cols: bool = False,
) -> pl.DataFrame:

    preds = model.predict(X_predict.to_pandas(), alpha=alpha)
    preds = np.concatenate(
        (preds[0].reshape(len(X_predict), -1), preds[1].reshape(len(X_predict), -1)),
        axis=1,
    )

    if pred_col is None:
        pred_col = "pred"
    lower_col = "lower_pred_interval"
    upper_col = "upper_pred_interval"

    preds = pl.DataFrame(
        preds,
        schema={
            pred_col: pl.Float64,
            lower_col: pl.Float64,
            upper_col: pl.Float64,
        },
    )
    if reverse_log_transform:
        preds = preds.with_columns(pl.col("*").exp())

    if include_interval_stats_cols:
        preds = preds.with_columns(
            [
                (pl.col(pred_col) - pl.col(lower_col)).alias("lower_pred_error"),
                (pl.col(upper_col) - pl.col(pred_col)).alias("upper_pred_error"),
                ((pl.col(upper_col) - pl.col(lower_col)) / 2).alias(
                    "avg_interval_width"
                ),
            ]
        ).with_columns(
            (pl.col("avg_interval_width") / pl.col(pred_col)).alias(
                "percent_interval_width"
            )
        )

    return preds


# creates arg to assign monotonic constraits for features
def assign_monotonic_csts(
    train_df: pl.DataFrame,
    increase_cols: Optional[List[str]] = None,
    decrease_cols: Optional[List[str]] = None,
) -> List[int]:

    if increase_cols is None:
        increase_cols = []
    if decrease_cols is None:
        decrease_cols = []

    common_cols = set(increase_cols).intersection(set(decrease_cols))
    if common_cols:
        raise ValueError(
            f"""Found common columns {common_cols}.
                         Increasing and decreasing columns should be mutually exclusive.
                         """
        )

    monotonic_cst = []
    for col in train_df.columns:
        if col in increase_cols:
            monotonic_cst.append(1)
        elif col in decrease_cols:
            monotonic_cst.append(-1)
        else:
            monotonic_cst.append(0)

    return monotonic_cst
