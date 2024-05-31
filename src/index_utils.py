from pathlib import Path

import polars as pl


def get_null_counts(df_pl: pl.DataFrame) -> None:
    return df_pl.select(pl.all().is_null().sum()).to_dicts()[0]


def add_flag_to_filepath(path: Path, flag: str) -> Path:
    # Add a flag to the filename and return path

    filename_parts = path.stem.split(".")
    filename_ext = path.suffix

    filename_parts.append(flag)

    output_filename = "_".join(filename_parts)
    output_filepath = path.parent / f"{output_filename}{filename_ext}"

    return output_filepath


def get_percentile_rank_pl(df_pl, col, descending=False, output_suffix=None):
    """Rank column values by percentile rank.
    Set descending=False to rank in reverse order.
    df_pl must be a polars dataframe (not pandas)"""

    values = df_pl[col]
    ranked_values = values.rank(method="dense")
    N = ranked_values.max()
    p_ranked_values = (ranked_values - 1) / (N - 1)

    if output_suffix is not None:
        p_ranked_values = p_ranked_values.alias(f"{values.name}{output_suffix}")

    return p_ranked_values


def calculate_svi(
    df_pl: pl.DataFrame,
    positive_features: list,
    negative_features: list,
    id_col: str = "quadkey",
    extreme_cutoff: float = 0.9,
    weights_dict: dict = None,
    return_all_features: bool = False,
):
    "Calculates the SVI based on percentile ranking"

    # If a weights dictionary is provided, assert that all features are present in dict
    if weights_dict is not None:
        all_features = positive_features + negative_features
        assert all(
            f in weights_dict for f in all_features
        ), "Not all entries are found in the dictionary keys"

    df_pl = df_pl.clone()
    output_df = df_pl.select(id_col)

    # get the percentile ranking of all features
    for f in positive_features:
        output_df = output_df.with_columns(
            get_percentile_rank_pl(df_pl, f, output_suffix="_p_rank")
        )
    for f in negative_features:
        output_df = output_df.with_columns(
            get_percentile_rank_pl(df_pl, f, output_suffix="_p_rank", descending=True)
        )

    # Get the columns with '_p_rank' suffix
    p_rank_cols = [col for col in output_df.columns if col.endswith("_p_rank")]

    # Filter the DataFrame to select only columns with '_p_rank' suffix
    p_rank_df = output_df.select(p_rank_cols)
    # Calculate SVI as mean of all percentile ranks
    # NaNs are converted to null so they can be ignored in mean
    if weights_dict is not None:
        weighted_p_rank_df = p_rank_df.clone()
        for f in all_features:
            weighted_p_rank_df = weighted_p_rank_df.with_columns(
                (pl.col(f"{f}_p_rank") * weights_dict[f]).alias(f"{f}_p_rank")
            )
            svi_mean_p_rank = (
                weighted_p_rank_df.fill_nan(None)
                .mean_horizontal(ignore_nulls=True)
                .alias("svi_mean_p_rank")
            )
    else:
        svi_mean_p_rank = (
            p_rank_df.fill_nan(None)
            .mean_horizontal(ignore_nulls=True)
            .alias("svi_mean_p_rank")
        )

    # Calculate SVI as total number of columns where p_rank is greater than cutoff
    svi_extreme_count = (
        (p_rank_df >= extreme_cutoff).sum_horizontal().alias("svi_extreme_count")
    )

    output_df = output_df.with_columns(svi_mean_p_rank, svi_extreme_count)
    if not return_all_features:
        output_df = output_df.select([id_col, "svi_mean_p_rank", "svi_extreme_count"])

    return output_df


def impute_nulls_by_group_median(
    df_pl: pl.DataFrame, cols_to_impute: list, group_col: str
) -> pl.DataFrame:
    df_pl = df_pl.clone()

    for col in cols_to_impute:
        df_pl = df_pl.with_columns(
            pl.col(col).fill_null(pl.col(col).median().over(group_col))
        )

    return df_pl
