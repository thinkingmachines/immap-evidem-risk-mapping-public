from typing import List, Optional

import numpy as np
import polars as pl
from loguru import logger
from scipy.spatial import distance


def log_condition(
    df: pl.DataFrame,
    cond: pl.Expr,
    n_rows_show: Optional[int] = None,
) -> pl.DataFrame:

    affected_df = df.filter(cond)
    n_affected = affected_df.height
    logger.info(f"There are {n_affected:,} rows meeting condition {cond}")

    if n_rows_show is not None:
        if (n_rows_show > n_affected) or (n_rows_show == -1):
            n_rows_str = "all"
            n_head = n_affected
        else:
            n_rows_str = n_rows_show
            n_head = n_rows_show

        logger.info(
            f"Showing {n_rows_str} rows that by condition {cond}\n{affected_df.head(n_head)}"
        )

    return df


def log_duplicates(
    df: pl.DataFrame,
    cols: Optional[List] = None,
    n_rows_show: Optional[int] = None,
) -> pl.DataFrame:

    if cols is None:
        usecols = df.columns
    else:
        usecols = cols

    affected_df = df.group_by(usecols).count().filter(pl.col("count") > 1)
    n_affected = affected_df.height

    if cols is None:
        logger.info(f"There are {n_affected:,} pure duplicate rows")
    else:
        logger.info(
            f"There are {n_affected:,} duplicate rows based on these columns {cols}"
        )

    if n_rows_show is not None:
        if (n_rows_show > n_affected) or (n_rows_show == -1):
            n_rows_str = "all"
            n_head = n_affected
        else:
            n_rows_str = n_rows_show
            n_head = n_rows_show

        logger.info(
            f"Showing {n_rows_str} rows that are duplicates\n{affected_df.head(n_head)}"
        )

    return df


def find_nearest_neighbors(
    df: pl.DataFrame,
    id_col: str,
    dist_cols: List[str],
    top_n: Optional[int] = None,
) -> pl.DataFrame:

    points = df.select(dist_cols).to_numpy()
    ids = df[id_col].to_numpy()

    dist_matrix = distance.cdist(points, points, "euclidean")

    # Create a large DataFrame with all distances and corresponding source and target IDs
    num_points = len(ids)
    source_ids = np.repeat(ids, num_points)
    target_ids = np.tile(ids, num_points)
    distances = dist_matrix.ravel()

    source_id_col = f"source_{id_col}"
    target_id_col = f"target_{id_col}"
    distances_df = pl.DataFrame(
        {
            source_id_col: source_ids,
            target_id_col: target_ids,
            "distance": distances,
        }
    )

    # Calculate ranks within each source_id group
    distances_df = (
        distances_df
        # Remove self-distances
        .filter(pl.col(source_id_col) != pl.col(target_id_col))
        .sort([source_id_col, "distance"])
        .with_columns(
            [
                pl.col("distance")
                .rank(method="dense")
                .over(source_id_col)
                .alias("distance_rank")
            ]
        )
    )

    if top_n is not None:
        distances_df = distances_df.filter(pl.col("distance_rank") <= top_n)

    return distances_df


def distances_df_join(
    distances_df: pl.DataFrame,
    other_df: pl.DataFrame,
    id_col: str,
) -> pl.DataFrame:

    assert id_col in other_df.columns

    source_other_df = other_df.clone()
    source_other_df.columns = [f"source_{col}" for col in other_df.columns]

    source_id_col = f"source_{id_col}"
    assert source_id_col in distances_df.columns
    distances_df = distances_df.join(
        source_other_df,
        on=source_id_col,
        how="left",
        validate="m:1",
    )

    target_other_df = other_df.clone()
    target_other_df.columns = [f"target_{col}" for col in other_df.columns]

    target_id_col = f"target_{id_col}"
    assert target_id_col in distances_df.columns
    distances_df = distances_df.join(
        target_other_df,
        on=target_id_col,
        how="left",
        validate="m:1",
    )

    return distances_df
