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
# # Create Training Data
#
# In this noteboook, our goal is to label each of our 150x150m grids to one of the three classes: landslide, flows, and non_landslide. Make sure to download the necssary input files and match them to the directories listed below.
#
# ### Inputs
# - Master feature table
# - Landslide coordinates from the landslide inventory
# - Non-landslide samples
#
# ### Process
# - Spatial intersectiono of the coordinates/polygons to add labels to each qaudkey
#
# ### Output
# - Tranining data with quadkey, features and labels

# %% [markdown]
# # Imports and Set-up
#
# *DO NOT SKIP THIS SECTION.* This section imports the packages needed to run this notebook and initializes the data file paths.

# %%
# %load_ext autoreload
# %autoreload 2

# Standard imports
import sys

# geospatial
import geopandas as gpd
import pandas as pd

# %%
# Util imports
sys.path.append("../../")  # include parent directory
from src.settings import DATA_DIR, LOCAL_CRS, PROJ_CRS

# %%
ALINGED_DIR = DATA_DIR / "aligned/csv"
MUNICIPALITIES_DIR = DATA_DIR / "admin_bounds"
VECTOR_DIR = DATA_DIR / "vectors"
TRAINING_DIR = VECTOR_DIR / "training_labels"
TRAINING_DIR.mkdir(exist_ok=True)

MASTERFILE_VERSION = "20240213"

dtype = {"quadkey": str}

AOI_FPATH = MUNICIPALITIES_DIR / "grids_landslide_w_xyz_zoomlevel18_20240320.parquet"
FEATURES_FPATH = ALINGED_DIR / "aligned_dataset_consolidated_20240507.csv"
IMMAP_DATA_FPATH = DATA_DIR / "vectors/client-data/cleaned_data"
IMMAP_DATA_FPATH.mkdir(exist_ok=True, parents=True)
MASTERFILE_FPATH = (
    IMMAP_DATA_FPATH / f"landslide_event_reference_file_{MASTERFILE_VERSION}.csv"
)

SAM_POLYGONS_DIR = DATA_DIR / "vectors/sam-outputs"
SAM_POLYGONS_DIR.mkdir(exist_ok=True)
SAM_POLYGONS_FPATH = SAM_POLYGONS_DIR / "sam_consolidated_w_qa20240213.gpkg"


NEG_LABEL_VERSION = "20240402"
NEG_LABEL_EXC_BUFFER_SIZE = 500
NEG_LABELS_FPATH = (
    TRAINING_DIR
    / f"non_landslide_sampled_grids_{NEG_LABEL_EXC_BUFFER_SIZE}m_{NEG_LABEL_VERSION}.gpkg"
)

VERSION = pd.to_datetime("today").strftime("%Y%m%d")
OUT_CSV_FPATH = DATA_DIR / f"models/training_data/training_data_{VERSION}.csv"
OUT_PARQUET_FPATH = DATA_DIR / f"models/training_data/training_data_{VERSION}.parquet"

REDUCE_LANDSLIDES = False  # set to True to create training data for Flows

# %% [markdown]
# ## Load Data

# %% [markdown]
# ### Labels
#
# This section loads both the positive and negative labels. For the positive labels, this will come from the masterfile (combination of landslide coordinates from Catalog and Inventory). For negative labels, this was generated beforehand using the buffer approach method.

# %%
masterfile = pd.read_csv(MASTERFILE_FPATH)
neg_labels_gdf = gpd.read_file(NEG_LABELS_FPATH)

# %%
masterfile.describe(), masterfile.info()

# %%
masterfile["qa"].value_counts(dropna=False)

# %%
# filter out coordinates that were part of SAM pipeline
masterfile = masterfile[masterfile["qa"].isna()]

# %%
masterfile.columns

# %%
masterfile.drop(
    columns=[
        "Unnamed: 0",
        "DPTO",
        "MUNICIPIO",
        "VEREDA",
        "Comments/Actions",
        "comments",
        "qa",
        "notes",
    ],
    inplace=True,
)

# %% [markdown]
# #### Filter to events 2000 onwards and to landslide, flows and topple

# %%
# format date
masterfile["MOV_DATE"] = pd.to_datetime(masterfile.MOV_DATE, errors="coerce")

# %%
pos_label = masterfile[
    (masterfile.MOV_DATE >= pd.to_datetime("20000101", format="%Y%m%d"))
    & (masterfile.MOV_TYPE.isin(["flows", "landslide"]))
]

# %%
pos_label.shape

# %%
pos_label.groupby(["MOV_TYPE", "source"]).count()

# %%
pos_labels_gdf = gpd.GeoDataFrame(
    pos_label,
    geometry=gpd.points_from_xy(pos_label.LON, pos_label.LAT),
    crs="EPSG:4326",
)

# %% [markdown]
# #### Check duplicate `OBJECTID`

# %%
pos_labels_gdf[pos_labels_gdf.duplicated(subset=["OBJECTID"], keep=False)]

# %%
pos_labels_gdf[pos_labels_gdf["OBJECTID"] == 94]

# %% [markdown]
# #### Check if any coordinates still overlap with SAM polygons

# %%
sam_polygons = gpd.read_file(SAM_POLYGONS_FPATH)
sam_polygons.head()

# %%
sam_polygons = sam_polygons[sam_polygons["qa"] == "pass"]

# %%
# first check for existing points that overlap with SAM polygons
check_overlaps = pos_labels_gdf.sjoin(sam_polygons)
check_overlaps

# %%
events_to_remove = check_overlaps["OBJECTID_left"].tolist()

# remove events that overlapped with SAM polygons
pos_labels_gdf = pos_labels_gdf[~pos_labels_gdf["OBJECTID"].isin(events_to_remove)]
pos_labels_gdf

# %% [markdown]
# ### AOI

# %%
AOI_FPATH

# %%
AOI_FPATH.exists()

# %%
USE_CACHED_GRIDS = True  # use cached grids if available

# %%
simple_aoi_grids = pd.read_parquet(AOI_FPATH)

# %%
simple_aoi_grids_geom = gpd.read_file(
    MUNICIPALITIES_DIR / "grids_landslide_wadm_zoomlevel18_20240304.gpkg"
)
simple_aoi_grids_geom = simple_aoi_grids_geom[["quadkey", "geometry"]]
simple_aoi_grids_geom

# %%
simple_aoi_grids = simple_aoi_grids.merge(simple_aoi_grids_geom, on="quadkey")
simple_aoi_grids = gpd.GeoDataFrame(simple_aoi_grids, geometry="geometry", crs=PROJ_CRS)
simple_aoi_grids

# %%
simple_aoi_grids = gpd.GeoDataFrame(simple_aoi_grids, geometry="geometry", crs=PROJ_CRS)
simple_aoi_grids

# %% [markdown]
# ### Feature table

# %%
features_df = pd.read_csv(FEATURES_FPATH, dtype=dtype)

# %% [markdown]
# # Spatial Joins
#
# Label the 150x150m grids either Landslide, Flows, and Non-landslide using a geospatial join.

# %% [markdown]
# ## Positive Labels

# %%
pos_labels_gdf.columns

# %%
# add the labels to grids
simple_aoi_grids.sjoin(pos_labels_gdf).shape

# %%
landslide_grids = simple_aoi_grids.sjoin(pos_labels_gdf)

# %%
# get all duplicates
duplicate_grids = landslide_grids[
    landslide_grids.duplicated(subset=["quadkey"], keep=False)
]


# %% [markdown]
# For this next code block, it looks at the duplicates and checks the data source. If a grid intersected with a point from both Catalog and Inventory we exclude that grid. The section belows outputs a list of quadkeys to exclude.


# %%
def get_inv_cat_dup(df):
    # Get quadkeys/grids that were tagged by both catalog and inventory
    # Group by quadkey and count occurrences of each landslide source
    grouped = df.groupby(by=["quadkey", "source"]).size().unstack(fill_value=0)
    # Filter quadkeys with duplicates in both 'catalog' and 'inventory'
    filtered_quadkeys = grouped[
        (grouped["landslide_catalog"] > 0) & (grouped["landslide_inventory"] > 0)
    ].index

    return filtered_quadkeys


# %%
# list of quadkeys to remove
remove_quad_cat = get_inv_cat_dup(duplicate_grids)

# remove from grids
landslide_grids = landslide_grids[~landslide_grids["quadkey"].isin(remove_quad_cat)]

# %%
# remove catalog
landslide_grids = landslide_grids[landslide_grids["source"] != "landslide_catalog"]

# %% [markdown]
# Th remaining grids are tagged by `landslide_inventory` at this point. Check the source column to make sure.

# %%
landslide_grids = landslide_grids.drop_duplicates(subset=["quadkey"], keep="first")

# %%
landslide_grids[landslide_grids.duplicated(subset=["quadkey"], keep=False)]

# %%
landslide_grids.OBJECTID.nunique()

# %%
landslide_grids.drop(
    columns=[
        "index_right",
        "OBJECTID",
        "MOV_DATE",
        "LAT",
        "LON",
        "geometry",
    ],
    inplace=True,
)

# %%
landslide_grids["MOV_TYPE"].value_counts()

# %%
if REDUCE_LANDSLIDES:
    # reduce landslide types
    # choose only quadkeys
    landslide_type_samples = landslide_grids[landslide_grids["MOV_TYPE"] == "landslide"]
    reduce_samples = int(landslide_grids.shape[0] * 0.30)
    selected_landslide_samples = landslide_type_samples.sample(
        n=reduce_samples, random_state=1
    )
    selected_landslide_quadkeys = selected_landslide_samples["quadkey"].tolist()
    # retain only flows and chosen quadkeys
    is_in_reduced_landslide = (
        landslide_grids["quadkey"].isin(selected_landslide_quadkeys)
    ) & (landslide_grids["MOV_TYPE"] == "landslide")
    is_flows = landslide_grids["MOV_TYPE"] == "flows"

    filtered_landslide_grids = landslide_grids[is_in_reduced_landslide | is_flows]
    landslide_grids = filtered_landslide_grids

# %%
landslide_grids["MOV_TYPE"].value_counts()

# %% [markdown]
# ## Negative Labels

# %%
neg_labels_gdf.shape

# %% [markdown]
# Further reduce the negative samples (non_landslide) to match the positive labels count as much as possible.

# %%
# reduce sample negative labels
reduce_samples = int(landslide_grids.shape[0] * 0.30)
neg_labels_gdf = neg_labels_gdf.sample(n=reduce_samples, random_state=1)

# %%
neg_labels_gdf = neg_labels_gdf.merge(
    simple_aoi_grids.drop(columns=["geometry"]), how="left"
)
neg_labels_gdf

# %%
neg_labels_gdf = neg_labels_gdf.reset_index()
neg_labels_gdf

# %%
neg_labels_gdf["MOV_TYPE"] = "non_landslide"

# %%
neg_labels_gdf["OBJECTID"] = neg_labels_gdf["quadkey"]

# %%
neg_labels_gdf.drop(columns=["geometry", "index"], inplace=True)

# %%
neg_labels_gdf.head(2)

# %% [markdown]
# ## Combine positive and negative labels into one dataframe

# %%
train_labels = pd.concat([landslide_grids, neg_labels_gdf], ignore_index=True)

# %%
train_labels["MOV_TYPE"].value_counts()

# %%
quadkey_dup = train_labels["quadkey"].duplicated()

train_labels[(quadkey_dup)]

# %% [markdown]
# # Append features to be used for training

# %%
train_data = train_labels.merge(features_df)

# %%
train_data.info()

# %%
train_data["OBJECTID"] = train_data["OBJECTID"].astype(str)

# %% [markdown]
# Drop the unnecessary administrative boundary columns.

# %%
train_data = train_data.drop(
    columns=[
        "MPIO_CRSLC",
        "MPIO_NAREA",
        "MPIO_NANO",
        "SHAPE_AREA",
        "SHAPE_LEN",
    ]
)

# %%
train_data

# %%
train_data.to_csv(OUT_CSV_FPATH, index=False)

# %%
train_data.to_parquet(OUT_PARQUET_FPATH, index=False)
