import geopandas as gpd
import pandas as pd

BLOCK_CENSUS_DATA_FIELD_MAP_EN = {
    "COD_DANE_A": "BLOCK_CODE",
    "MPIO_CDPMP": "MUNI_CODE",
    "VERSION": "VERSION",
    "AREA": "AREA_SQM",
    "LATITUD": "LAT",
    "LONGITUD": "LON",
    "DENSIDAD": "POP_DENSITY",
    "CTNENCUEST": "CNT_SURVEYS",
    "TP3_1_SI": "CNT_SURVEYS_ETH",
    "TP3_2_NO": "CNT_SURVEYS_NOT_ETH",
    "TP3A_RI": "CNT_SURVEYS_IND",
    "TP3B_TCN": "CNT_SURVEYS_BLACK",
    "TP4_1_SI": "CNT_SURVEYS_PROTECTED",
    "TP4_2_NO": "CNT_SURVEYS_NOT_PROTECTED",
    "TVIVIENDA": "CNT_DWELLINGS",
    "TP14_1_TIP": "CNT_HOUSES",
    "TP14_2_TIP": "CNT_APARTMENT",
    "TP14_3_TIP": "CNT_ROOM_TYPE",
    "TP14_4_TIP": "CNT_IND",
    "TP14_5_TIP": "CNT_ETH",
    "TP14_6_TIP": "CNT_NATURAL",
    "TP16_HOG": "CNT_HOUSEHOLD",
    "TP19_EE_1": "CNT_W_ELEC",
    "TP19_EE_2": "CNT_WO_ELEC",
    "TP19_EE_E1": "CNT_ELEC_STRATUM_1",
    "TP19_EE_E2": "CNT_ELEC_STRATUM_2",
    "TP19_EE_E3": "CNT_ELEC_STRATUM_3",
    "TP19_EE_E4": "CNT_ELEC_STRATUM_4",
    "TP19_EE_E5": "CNT_ELEC_STRATUM_5",
    "TP19_EE_E6": "CNT_ELEC_STRATUM_6",
    "TP19_EE_E9": "CNT_ELEC_STRATUM_UNKNOWN",
    "TP19_ACU_1": "CNT_WATER_SERVICE",
    "TP19_ACU_2": "CNT_NO_WATER_SERVICE",
    "TP19_ALC_1": "CNT_SEWERAGE_SERVICE",
    "TP19_ALC_2": "CNT_NO_SEWERAGE_SERVICE",
    "TP19_GAS_1": "CNT_NATURAL_GAS_CONNECTED",
    "TP19_GAS_2": "CNT_NO_NATURAL_GAS_CONNECTED",
    "TP19_GAS_9": "CNT_NATURAL_GAS_UNKNOWN",
    "TP19_RECB1": "CNT_GARBAGE_COLLECTION_SERVICE",
    "TP19_RECB2": "CNT_NO_GARBAGE_COLLECTION_SERVICE",
    "TP19_INTE1": "CNT_INTERNET_SERVICE",
    "TP19_INTE2": "CNT_NO_INTERNET_SERVICE",
    "TP19_INTE9": "CNT_INTERNET_UNKNOWN",
    "TP27_PERSO": "CNT_POPULATION_TOTAL",
    "TP32_1_SEX": "CNT_POPULATION_MEN",
    "TP32_2_SEX": "CNT_POPULATION_WOMEN",
    "TP34_1_EDA": "CNT_POPULATION_0_9",
    "TP34_2_EDA": "CNT_POPULATION_10_19",
    "TP34_3_EDA": "CNT_POPULATION_20_29",
    "TP34_4_EDA": "CNT_POPULATION_30_39",
    "TP34_5_EDA": "CNT_POPULATION_40_49",
    "TP34_6_EDA": "CNT_POPULATION_50_59",
    "TP34_7_EDA": "CNT_POPULATION_60_69",
    "TP34_8_EDA": "CNT_POPULATION_70_79",
    "TP34_9_EDA": "CNT_POPULATION_80_OVER",
    "TP51PRIMAR": "CNT_POPULATION_EDU_LEVEL_PRIMARY",
    "TP51SECUND": "CNT_POPULATION_EDU_LEVEL_SECONDARY",
    "TP51SUPERI": "CNT_POPULATION_EDU_LEVEL_TECHNICAL_PROFESSIONAL",
    "TP51POSTGR": "CNT_POPULATION_EDU_LEVEL_POSTGRADUATE",
    "TP51_13_ED": "CNT_POPULATION_EDU_LEVEL_NONE",
    "TP51_99_ED": "CNT_POPULATION_EDU_LEVEL_UNKNOWN",
}

RURAL_CENSUS_DATA_FIELD_MAP_EN = {
    "SECR_CCNCT": "RURAL_CODE",
    "SECU_CCNCT": "URBAN_SECTION_CODE",
    "VERSION": "VERSION",
    "AREA": "AREA_SQM",
    "LATITUD": "LAT",
    "LONGITUD": "LON",
    "STCTNENCUE": "CNT_SURVEYS",
    "STP3_1_SI": "CNT_SURVEYS_ETH",
    "STP3_2_NO": "CNT_SURVEYS_NOT_ETH",
    "STP3A_RI": "CNT_SURVEYS_IND",
    "STP3B_TCN": "CNT_SURVEYS_BLACK",
    "STP4_1_SI": "CNT_SURVEYS_PROTECTED",
    "STP4_2_NO": "CNT_SURVEYS_NOT_PROTECTED",
    "STVIVIENDA": "CNT_DWELLINGS",
    "STP14_1_TI": "CNT_HOUSES",
    "STP14_2_TI": "CNT_APARTMENT",
    "STP14_3_TI": "CNT_ROOM_TYPE",
    "STP14_4_TI": "CNT_IND",
    "STP14_5_TI": "CNT_ETH",
    "STP14_6_TI": "CNT_NATURAL",
    "TSP16_HOG": "CNT_HOUSEHOLD",
    "STP19_EC_1": "CNT_W_ELEC",
    "STP19_ES_2": "CNT_WO_ELEC",
    "STP19_EE_1": "CNT_ELEC_STRATUM_1",
    "STP19_EE_2": "CNT_ELEC_STRATUM_2",
    "STP19_EE_3": "CNT_ELEC_STRATUM_3",
    "STP19_EE_4": "CNT_ELEC_STRATUM_4",
    "STP19_EE_5": "CNT_ELEC_STRATUM_5",
    "STP19_EE_6": "CNT_ELEC_STRATUM_6",
    "STP19_EE_9": "CNT_ELEC_STRATUM_UNKNOWN",
    "STP19_ACU1": "CNT_WATER_SERVICE",
    "STP19_ACU2": "CNT_NO_WATER_SERVICE",
    "STP19_ALC1": "CNT_SEWERAGE_SERVICE",
    "STP19_ALC2": "CNT_NO_SEWERAGE_SERVICE",
    "STP19_GAS1": "CNT_NATURAL_GAS_CONNECTED",
    "STP19_GAS2": "CNT_NO_NATURAL_GAS_CONNECTED",
    "STP19_GAS9": "CNT_NATURAL_GAS_UNKNOWN",
    "STP19_REC1": "CNT_GARBAGE_COLLECTION_SERVICE",
    "STP19_REC2": "CNT_NO_GARBAGE_COLLECTION_SERVICE",
    "STP19_INT1": "CNT_INTERNET_SERVICE",
    "STP19_INT2": "CNT_NO_INTERNET_SERVICE",
    "STP19_INT9": "CNT_INTERNET_UNKNOWN",
    "STP27_PERS": "CNT_POPULATION_TOTAL",
    "STP32_1_SE": "CNT_POPULATION_MEN",
    "STP32_2_SE": "CNT_POPULATION_WOMEN",
    "STP34_1_ED": "CNT_POPULATION_0_9",
    "STP34_2_ED": "CNT_POPULATION_10_19",
    "STP34_3_ED": "CNT_POPULATION_20_29",
    "STP34_4_ED": "CNT_POPULATION_30_39",
    "STP34_5_ED": "CNT_POPULATION_40_49",
    "STP34_6_ED": "CNT_POPULATION_50_59",
    "STP34_7_ED": "CNT_POPULATION_60_69",
    "STP34_8_ED": "CNT_POPULATION_70_79",
    "STP34_9_ED": "CNT_POPULATION_80_OVER",
    "STP51_PRIM": "CNT_POPULATION_EDU_LEVEL_PRIMARY",
    "STP51_SECU": "CNT_POPULATION_EDU_LEVEL_SECONDARY",
    "STP51_SUPE": "CNT_POPULATION_EDU_LEVEL_TECHNICAL_PROFESSIONAL",
    "STP51_POST": "CNT_POPULATION_EDU_LEVEL_POSTGRADUATE",
    "STP51_13_E": "CNT_POPULATION_EDU_LEVEL_NONE",
    "STP51_99_E": "CNT_POPULATION_EDU_LEVEL_UNKNOWN",
}


FEATURES_MAPPING = {
    "CNT_DWELLINGS": "census_dwellings_count",
    "CNT_HOUSEHOLD": "census_household_count",
    "CNT_POPULATION_TOTAL": "census_population_total_count",
    "CNT_POPULATION_MEN": "census_population_men_count",
    "CNT_POPULATION_WOMEN": "census_population_women_count",
    "CNT_POPULATION_0_9": "census_population_0_9_count",
    "CNT_POPULATION_10_19": "census_population_10_19_count",
    "CNT_POPULATION_20_29": "census_population_20_29_count",
    "CNT_POPULATION_30_39": "census_population_30_39_count",
    "CNT_POPULATION_40_49": "census_population_40_49_count",
    "CNT_POPULATION_50_59": "census_population_50_59_count",
    "CNT_POPULATION_60_69": "census_population_60_69_count",
    "CNT_POPULATION_70_79": "census_population_70_79_count",
    "CNT_POPULATION_80_OVER": "census_population_80_over_count",
    "CNT_POPULATION_MEN_percent": "census_population_men_percent",
    "CNT_POPULATION_WOMEN_percent": "census_population_women_percent",
    "CNT_POPULATION_0_9_percent": "census_population_0_9_percent",
    "CNT_POPULATION_10_19_percent": "census_population_10_19_percent",
    "CNT_POPULATION_20_29_percent": "census_population_20_29_percent",
    "CNT_POPULATION_30_39_percent": "census_population_30_39_percent",
    "CNT_POPULATION_40_49_percent": "census_population_40_49_percent",
    "CNT_POPULATION_50_59_percent": "census_population_50_59_percent",
    "CNT_POPULATION_60_69_percent": "census_population_60_69_percent",
    "CNT_POPULATION_70_79_percent": "census_population_70_79_percent",
    "CNT_POPULATION_80_OVER_percent": "census_population_80_over_percent",
    "POP_DENSITY": "census_population_density_mean",
    "CNT_POPULATION_EDU_LEVEL_PRIMARY": "census_population_edu_level_primary_count",
    "CNT_POPULATION_EDU_LEVEL_SECONDARY": "census_population_edu_level_secondary_count",
    "CNT_POPULATION_EDU_LEVEL_TECHNICAL_PROFESSIONAL": "census_population_edu_level_technical_professional_count",
    "CNT_POPULATION_EDU_LEVEL_POSTGRADUATE": "census_population_edu_level_postgraduate_count",
    "CNT_POPULATION_EDU_LEVEL_NONE": "census_population_edu_level_none_count",
    "CNT_POPULATION_EDU_LEVEL_UNKNOWN": "census_population_edu_level_unknown_count",
    "CNT_POPULATION_EDU_LEVEL_PRIMARY_percent": "census_population_edu_level_primary_percent",
    "CNT_POPULATION_EDU_LEVEL_SECONDARY_percent": "census_population_edu_level_secondary_percent",
    "CNT_POPULATION_EDU_LEVEL_TECHNICAL_PROFESSIONAL_percent": "census_population_edu_level_technical_professional_percent",
    "CNT_POPULATION_EDU_LEVEL_POSTGRADUATE_percent": "census_population_edu_level_postgraduate_percent",
    "CNT_POPULATION_EDU_LEVEL_NONE_percent": "census_population_edu_level_none_percent",
    "CNT_POPULATION_EDU_LEVEL_UNKNOWN_percent": "census_population_edu_level_unknown_percent",
    "CNT_WATER_SERVICE": "census_dwellings_water_service_count",
    "CNT_NO_WATER_SERVICE": "census_dwellings_no_water_service_count",
    "CNT_SEWERAGE_SERVICE": "census_dwellings_sewerage_service_count",
    "CNT_NO_SEWERAGE_SERVICE": "census_dwellings_no_sewerage_service_count",
    "CNT_GARBAGE_COLLECTION_SERVICE": "census_dwellings_garbage_collection_service_count",
    "CNT_NO_GARBAGE_COLLECTION_SERVICE": "census_dwellings_no_garbage_collection_service_count",
    "CNT_WATER_SERVICE_percent": "census_dwellings_water_service_percent",
    "CNT_NO_WATER_SERVICE_percent": "census_dwellings_no_water_service_percent",
    "CNT_SEWERAGE_SERVICE_percent": "census_dwellings_sewerage_service_percent",
    "CNT_NO_SEWERAGE_SERVICE_percent": "census_dwellings_no_sewerage_service_percent",
    "CNT_GARBAGE_COLLECTION_SERVICE_percent": "census_dwellings_garbage_collection_service_percent",
    "CNT_NO_GARBAGE_COLLECTION_SERVICE_percent": "census_dwellings_no_garbage_collection_service_percent",
    "CNT_W_ELEC": "census_dwellings_w_elec_count",
    "CNT_WO_ELEC": "census_dwellings_wo_elec_count",
    "CNT_W_ELEC_percent": "census_dwellings_w_elec_percent",
    "CNT_WO_ELEC_percent": "census_dwellings_wo_elec_percent",
    "CNT_IND": "census_dwellings_ind_count",
    "CNT_ETH": "census_dwellings_eth_count",
    "CNT_IND_percent": "census_dwellings_ind_percent",
    "CNT_ETH_percent": "census_dwellings_eth_percent",
    "CNT_INTERNET_SERVICE": "census_dwellings_internet_service_count",
    "CNT_NO_INTERNET_SERVICE": "census_dwellings_no_internet_service_count",
    "CNT_INTERNET_UNKNOWN": "census_dwellings_internet_unknown_count",
    "CNT_INTERNET_SERVICE_percent": "census_dwellings_internet_service_percent",
    "CNT_NO_INTERNET_SERVICE_percent": "census_dwellings_no_internet_service_percent",
    "CNT_INTERNET_UNKNOWN_percent": "census_dwellings_internet_unknown_percent",
    "CNT_POPULATION_DEPENDENT": "census_population_dependent_count",
    "CNT_POPULATION_DEPENDENT_percent": "census_population_dependent_percent",
    "CNT_POPULATION_EDU_LEVEL_TERTIARY": "census_population_edu_level_tertiary_count",
    "CNT_POPULATION_EDU_LEVEL_TERTIARY_percent": "census_population_edu_level_tertiary_percent",
}

# All the target features have CNT in name
CENSUS_TARGET_FEATURES = list(
    dict.fromkeys(
        [
            x
            for x in list(BLOCK_CENSUS_DATA_FIELD_MAP_EN.values())
            + list(RURAL_CENSUS_DATA_FIELD_MAP_EN.values())
            if "CNT" in x
        ]
    )
)


def check_same_columns(*dfs):
    """Helper function to check if dataframes have the same columns.
    Usage: check_same_columns(df1, df2, df3, ...)"""
    cols_sets = [set(df.columns) for df in dfs]

    if all(cols == cols_sets[0] for cols in cols_sets):
        return True
    else:
        return False


def map_field_to_gdf(gdf, field_map, mode="replace"):
    gdf = gdf.copy()

    assert mode in ["replace", "append"], "mode not set to either `replace` or `append`"
    if mode == "replace":
        gdf = gdf.rename(field_map, axis=1)
    elif mode == "append":
        for input_name, output_name in field_map.items():
            gdf[output_name] = gdf[input_name]

    return gdf


def calculate_percent_features(gdf, cols, denominator_col, drop_denominator=False):
    percent_output = gdf.copy()
    percent_output = percent_output.loc[:, ["quadkey", denominator_col] + cols]

    for col in cols:
        percent_output[f"{col}_percent"] = (
            percent_output[col] / percent_output[denominator_col]
        )

    if drop_denominator:
        percent_output = percent_output.drop(denominator_col, axis=1)

    return percent_output


def calculate_dependent_total(gdf):
    output = gdf.copy()
    DEPENDENT_COLS = [
        "CNT_POPULATION_0_9",
        "CNT_POPULATION_60_69",
        "CNT_POPULATION_70_79",
        "CNT_POPULATION_80_OVER",
    ]
    output["CNT_POPULATION_DEPENDENT"] = output[DEPENDENT_COLS].sum(axis=1)
    return output


def calculate_tertiary_total(gdf):
    output = gdf.copy()
    TERTIARY_COLS = [
        "CNT_POPULATION_EDU_LEVEL_TECHNICAL_PROFESSIONAL",
        "CNT_POPULATION_EDU_LEVEL_POSTGRADUATE",
    ]
    output["CNT_POPULATION_EDU_LEVEL_TERTIARY"] = output[TERTIARY_COLS].sum(axis=1)
    return output


## Depreciated functions from first iteration
## We keep this for now so that the old notebooks won't be broken


def get_sum_groupby_quadkey(block_gdf, sum_cols):
    "Get the sum of the specified cols per quadkey"
    block_gdf = block_gdf.copy()
    sum_output = (
        block_gdf[["quadkey"] + sum_cols]
        .groupby("quadkey")
        .sum()
        .reset_index(drop=False)
    )

    return sum_output


def get_mean_groupby_quadkey(block_gdf, mean_cols):
    "Get the mean of the specified cols per quadkey"
    block_gdf = block_gdf.copy()
    sum_output = (
        block_gdf[["quadkey"] + mean_cols]
        .groupby("quadkey")
        .mean()
        .reset_index(drop=False)
    )

    return sum_output


def get_sum_and_percent_gropupby_quadkey(
    block_gdf, cols, denominator_col, drop_denominator=False
):
    "Get the sum and percent of specified cols per quadkey. The denominator col must also be in cols"
    assert denominator_col in cols, f"{denominator_col} not in passed columns"

    block_gdf = block_gdf.copy()
    output = get_sum_groupby_quadkey(block_gdf, cols)
    for col in cols:
        if col == denominator_col:
            continue
        output[f"{col}_percent"] = output[col] / output[denominator_col]
    if drop_denominator:
        output = output.drop(denominator_col, axis=1)

    return output


def get_dependent_sum_and_percent(block_gdf, drop_total_population_col=True):
    "Helper function to get sum and percent of dependent population (default: under 10 y/o and above 60 y/0)"
    DEPENDENT_COLS = [
        "CNT_POPULATION_0_9",
        "CNT_POPULATION_60_69",
        "CNT_POPULATION_70_79",
        "CNT_POPULATION_80_OVER",
    ]
    POPULATION_COL = "CNT_POPULATION_TOTAL"

    dependent_block_gdf = block_gdf.copy()
    dependent_block_gdf["CNT_POPULATION_DEPENDENT"] = dependent_block_gdf[
        DEPENDENT_COLS
    ].sum(axis=1)
    output = get_sum_and_percent_gropupby_quadkey(
        dependent_block_gdf,
        ["CNT_POPULATION_DEPENDENT", "CNT_POPULATION_TOTAL"],
        POPULATION_COL,
        drop_denominator=drop_total_population_col,
    )

    return output


def get_tertiary_education_sum_and_percent(block_gdf, drop_total_population_col=True):
    "Helper function to get sum and percent of tertiary education population (default: sum of tertiary and post-graduate)"
    TERTIARY_COLS = [
        "CNT_POPULATION_EDU_LEVEL_TECHNICAL_PROFESSIONAL",
        "CNT_POPULATION_EDU_LEVEL_POSTGRADUATE",
    ]
    POPULATION_COL = "CNT_POPULATION_TOTAL"

    dependent_block_gdf = block_gdf.copy()
    dependent_block_gdf["CNT_POPULATION_EDU_LEVEL_TERTIARY"] = dependent_block_gdf[
        TERTIARY_COLS
    ].sum(axis=1)
    output = get_sum_and_percent_gropupby_quadkey(
        dependent_block_gdf,
        ["CNT_POPULATION_EDU_LEVEL_TERTIARY", "CNT_POPULATION_TOTAL"],
        POPULATION_COL,
        drop_denominator=drop_total_population_col,
    )

    return output
