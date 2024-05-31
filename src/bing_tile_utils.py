from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import polars as pl

CIRCUMFERENCE_EARTH_KM = 40_075
CHEBYSHEV_DIST_COLS = ["chebyshev_dist", "chebyshev_pseudo_dist"]


def get_bing_tile_length_m(zoom_level: int) -> float:
    assert isinstance(zoom_level, int) and zoom_level > 0

    num_tiles = 2**zoom_level
    tile_length_km = CIRCUMFERENCE_EARTH_KM / num_tiles
    tile_length_m = tile_length_km * 1000
    return tile_length_m


def get_bing_cluster_tile_length_m(zoom_level: int, radius: int) -> float:
    assert isinstance(radius, int) and radius >= 0

    tile_length_m = get_bing_tile_length_m(zoom_level)
    n_tiles_row = 2 * radius + 1
    cluster_length_m = tile_length_m * n_tiles_row
    return cluster_length_m


def generate_lattice_at_origin(
    radius: int,
    include_chebyshev_dist: bool = False,
    return_lazyframe: bool = False,
) -> pl.DataFrame:
    assert isinstance(radius, int) and radius >= 0
    assert radius < 127, f"radius {radius} is out of range"

    x = pl.int_range(-radius, radius + 1, eager=True)
    y = pl.int_range(-radius, radius + 1, eager=True)

    x = pl.DataFrame(x, schema={"x": pl.Int8})
    y = pl.DataFrame(y, schema={"y": pl.Int8})

    if return_lazyframe:
        x = x.lazy()
        y = y.lazy()

    lattice = x.join(y, how="cross")
    if include_chebyshev_dist:
        expr1 = pl.max_horizontal(pl.col("x").abs(), pl.col("y").abs()).alias(
            "chebyshev_dist"
        )

        lattice = lattice.with_columns(expr1)

        expr2 = (
            pl.when((pl.col("chebyshev_dist") == 0) & (radius > 0))
            .then(pl.lit(1))
            .otherwise(pl.col("chebyshev_dist"))
            .alias("chebyshev_pseudo_dist")
        )
        lattice = lattice.with_columns(expr2)

    return lattice


def generate_lattice(
    centers_df: pl.DataFrame,
    radius: int,
    include_centers: bool = True,
    zoom_level: Optional[int] = None,
    include_chebyshev_dist: bool = False,
) -> pl.DataFrame:

    assert list(centers_df.columns) == ["x", "y"]
    return_lazyframe = isinstance(centers_df, pl.LazyFrame)

    # shape of lattice_origin_df should be is ((2*radius+1)**2, 2)
    lattice_origin_df = generate_lattice_at_origin(
        radius=radius,
        return_lazyframe=return_lazyframe,
        include_chebyshev_dist=include_chebyshev_dist,
    )
    lattice_origin_df = lattice_origin_df.rename(
        {"x": "lattice_origin_x", "y": "lattice_origin_y"}
    )

    # shift the lattice from the origin (0,0) to where the centers are
    centers_df = centers_df.rename({"x": "center_x", "y": "center_y"})
    lattice_df = centers_df.join(lattice_origin_df, how="cross")
    lattice_df = lattice_df.with_columns(
        [
            (pl.col("lattice_origin_x") + pl.col("center_x")).alias("lattice_x"),
            (pl.col("lattice_origin_y") + pl.col("center_y")).alias("lattice_y"),
        ]
    )

    lattice_cols = ["center_x", "center_y", "lattice_x", "lattice_y"]

    if include_chebyshev_dist:
        chebyshev_cols = ["chebyshev_dist", "chebyshev_pseudo_dist"]
        lattice_cols += chebyshev_cols
    lattice_df = lattice_df.select(lattice_cols)

    if not include_centers:
        # Filter out the center points
        centers_expr = (pl.col("center_x") == pl.col("lattice_x")) & (
            pl.col("center_y") == pl.col("lattice_y")
        )
        lattice_df = lattice_df.filter(~centers_expr)

    if zoom_level is not None:
        quadkey_expr = xyz_to_quadkey_expr(
            pl.col("lattice_x"), pl.col("lattice_y"), zoom_level
        )
        quadkey_expr = quadkey_expr.alias("lattice_quadkey")

        # compute the quadkey on the unique lattice xy so we save on compute
        unique_lattice = (
            lattice_df.select(["lattice_x", "lattice_y"])
            .unique()
            .with_columns(quadkey_expr)
        )
        lattice_df = lattice_df.join(
            unique_lattice, how="left", on=["lattice_x", "lattice_y"]
        )

    return lattice_df


def xyz_to_quadkey_expr(x: pl.Expr, y: pl.Expr, zoom: int) -> pl.Expr:

    # Create expressions for the quadkey digit at each bit position
    quadkey_digit_exprs = [
        ((x // (2**i) % 2) | ((y // (2**i) % 2) * 2)) for i in reversed(range(zoom))
    ]

    quadkey = pl.concat_str(quadkey_digit_exprs)

    return quadkey


def get_chebyshev_count_exprs(
    group_by_cols: Union[str, List[str]],
    chebyshev_dist_col: str = "chebyshev_pseudo_dist",
) -> List[pl.Expr]:
    if isinstance(group_by_cols, str):
        group_by_cols = [group_by_cols]

    lattice_count_expr = pl.len().over(group_by_cols).alias("lattice_quadkey_count")

    over_cols = [chebyshev_dist_col] + group_by_cols
    level_count_expr = pl.len().over(over_cols).alias("chebyshev_level_count")

    chebyshev_count_exprs = [level_count_expr, lattice_count_expr]
    return chebyshev_count_exprs


def get_lattice_weight_expr(
    radius: int,
    group_by_cols: Union[str, List[str]],
    level_count_col: str,
    chebyshev_dist_col: str = "chebyshev_dist",
) -> pl.Expr:

    # tiles closer to the center are given higher weight
    # the weights decrease per level by a factor of 2
    # caveat: depending on the defintion of level_count_column
    #    levels 0 and 1 could be counted as one group since level 0 just has 1 tile
    unnormalized_weight_expr = 2 ** (radius - pl.col(chebyshev_dist_col)) / pl.col(
        level_count_col
    )

    # normalize the probabilities by making sure they add to 100% per center quadkey
    normalized_weight_expr = (
        unnormalized_weight_expr / unnormalized_weight_expr.sum().over(group_by_cols)
    )
    normalized_weight_expr = normalized_weight_expr.alias("lattice_weight")

    return normalized_weight_expr


def get_lattice_weight_exprs(
    use_weighted_lattice: bool,
    radius: int,
    group_by_cols: Union[List[str], str],
    chebyshev_dist_col: str = "chebyshev_pseudo_dist",
) -> Dict[str, Any]:

    # Convenience function that returns relevant expressions if doing weighted lattice
    if use_weighted_lattice:
        chebyshev_count_exprs = get_chebyshev_count_exprs(
            group_by_cols=group_by_cols, chebyshev_dist_col=chebyshev_dist_col
        )

        level_count_col = "chebyshev_level_count"
        lattice_weight_expr = get_lattice_weight_expr(
            radius=radius,
            group_by_cols=group_by_cols,
            level_count_col=level_count_col,
        )
        lattice_weight_exprs = [
            (lattice_weight_expr * pl.col("lattice_quadkey_count")).alias(
                "lattice_weight"
            )
        ]
        lattice_weight_multiplier = pl.col("lattice_weight")

    else:
        # list of empty lists will not make new columns
        chebyshev_count_exprs = []
        lattice_weight_exprs = []
        lattice_weight_multiplier = 1

    exprs_dict = {
        "chebyshev_count_exprs": chebyshev_count_exprs,
        "lattice_weight_exprs": lattice_weight_exprs,
        "lattice_weight_multiplier": lattice_weight_multiplier,
    }

    return exprs_dict


def generate_lattice_at_origin_bigquery(radius: int) -> str:
    query = f"""
    DECLARE n INT64 DEFAULT {radius};

    WITH
    numbers AS (
        SELECT number
        FROM UNNEST(GENERATE_ARRAY(-n, n, 1)) AS number
    ),

    lattice AS (
        SELECT n1.number AS x, n2.number AS y
        FROM numbers n1
        CROSS JOIN numbers n2
    )

    SELECT * FROM lattice
    """

    return query


def generate_lattice_bigquery(
    centers_table: str, radius: int, include_centers: bool = True
) -> str:
    query = f"""
    DECLARE n INT64 DEFAULT {radius};

    WITH
    numbers AS (
        SELECT number
        FROM UNNEST(GENERATE_ARRAY(-n, n, 1)) AS number
    ),

    lattice AS (
        SELECT n1.number AS x, n2.number AS y
        FROM numbers n1
        CROSS JOIN numbers n2
    ),

    centers AS (
        SELECT
            x as center_x,
            y as center_y
        FROM `{centers_table}`
    ),

    shiftedlattice AS (
        SELECT
            c.center_x,
            c.center_y,
            c.center_x + l.x AS lattice_x,
            c.center_y + l.y AS lattice_y,

        FROM lattice l
        CROSS JOIN centers c
    )

    SELECT * FROM shiftedlattice
    """

    if not include_centers:
        query = f"""
        {query}

        WHERE
            (center_x != lattice_x)
            AND (center_y != lattice_y)
        """

    return query


def generate_lattice_at_origin_numpy(radius: int) -> np.array:
    assert isinstance(radius, int) and radius >= 0

    x = np.arange(-radius, radius + 1)
    y = np.arange(-radius, radius + 1)

    meshgrid = np.meshgrid(x, y)
    lattice = np.array(meshgrid).T.reshape(-1, 2)
    return lattice


def xyz_to_quadkey_numpy(x: np.array, y: np.array, zoom: int) -> np.array:
    i = np.arange(zoom)[::-1]
    x_digits = x[:, None] // (2**i) % 2
    y_digits = y[:, None] // (2**i) % 2
    quadkey_digits = (x_digits | (y_digits * 2)).astype(str)
    quadkeys = np.apply_along_axis("".join, 1, quadkey_digits)
    return quadkeys


def generate_lattice_pandas(
    centers_df: pd.DataFrame,
    radius: int,
    include_centers: bool = True,
    zoom_level: Optional[int] = None,
) -> pd.DataFrame:

    assert list(centers_df.columns) == ["x", "y"]

    # shape is ((2*radius+1)**2, 2)
    lattice_origin_points = generate_lattice_at_origin_numpy(radius)

    # Reshape center coordinates for broadcasting
    # shape is (num_centers, 1, 2)
    center_coords = centers_df.to_numpy()[:, np.newaxis, :]

    # Shift the lattice from the origin (0,0) to where the centers are
    combined_points = lattice_origin_points + center_coords

    combined_points_reshaped = combined_points.reshape(-1, 2)
    lattice_df = pd.DataFrame(
        combined_points_reshaped, columns=["lattice_x", "lattice_y"]
    )
    lattice_df[["center_x", "center_y"]] = np.repeat(
        centers_df.values, len(lattice_origin_points), axis=0
    )

    usecols = ["center_x", "center_y", "lattice_x", "lattice_y"]
    lattice_df = lattice_df[usecols]

    if not include_centers:
        bool_mask = lattice_df["center_x"] == lattice_df["lattice_x"]
        bool_mask = bool_mask & (lattice_df["center_y"] == lattice_df["lattice_y"])

        lattice_df = lattice_df.loc[~bool_mask, :]

    if zoom_level is not None:

        # compute the quadkey on the unique lattice xy so we save on compute
        unique_lattice_df = lattice_df[["lattice_x", "lattice_y"]].drop_duplicates()
        unique_lattice_df["lattice_quadkey"] = xyz_to_quadkey_numpy(
            unique_lattice_df["lattice_x"].to_numpy(),
            unique_lattice_df["lattice_y"].to_numpy(),
            zoom_level,
        )

        lattice_df = pd.merge(
            left=lattice_df,
            right=unique_lattice_df,
            on=["lattice_x", "lattice_y"],
            how="left",
        )

    return lattice_df
