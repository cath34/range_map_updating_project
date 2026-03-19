import os
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from rasterio.transform import xy

import raster_utils
import baseline_merow


species = "bird_2"
dataset_type = "full"
output_dir = "Results"


######

# Step 1 : Build 27km2 grid from AOI and convert it into a raster format

######

cell_area_km2 = 27.0

aoi_shp = "Dataset/range_maps/aoi/alaska_area.shp"
aoi_for_crs = gpd.read_file(aoi_shp)
target_crs = aoi_for_crs.estimate_utm_crs()

cell_size_m = np.sqrt(cell_area_km2) * 1000.0

aoi_gdf, transform, width, height = raster_utils.build_target_grid_from_aoi(
    aoi_shp=aoi_shp, target_crs=target_crs, cell_size_m=cell_size_m
)
print(f"Grid width={width}, height={height}")

aoi_mask_arr = raster_utils.rasterize_polygon_to_grid(
    gdf=aoi_gdf, transform=transform, width=width, height=height
)
aoi_mask = aoi_mask_arr.astype(bool)

######

# Step 2 : Build Expert Range Map raster from AOI grid

######

expert_shp = f"Dataset/range_maps/experts/{species}/{species}.shp"
expert_gdf = gpd.read_file(expert_shp).to_crs(target_crs)

expert_mask_arr = raster_utils.rasterize_polygon_to_grid(
    gdf=expert_gdf, transform=transform, width=width, height=height
)
expert_mask = expert_mask_arr.astype(bool)

raster_utils.save_array_as_raster(
    path=os.path.join(output_dir, f"rasterized_expert_maps/{species}.tif"),
    arr=expert_mask_arr,
    transform=transform,
    crs=target_crs,
    nodata=0,
    dtype="uint8",
)

######

# Step 3 : Resample covariates to AOI grid

######

resampling_by_raster = {
    "Dataset/covariates/raw/bio_1.tif": "bilinear",
    "Dataset/covariates/raw/bio_2.tif": "bilinear",
    "Dataset/covariates/raw/bio_3.tif": "bilinear",
    "Dataset/covariates/raw/bio_4.tif": "bilinear",
    "Dataset/covariates/raw/bio_5.tif": "bilinear",
    "Dataset/covariates/raw/bio_6.tif": "bilinear",
    "Dataset/covariates/raw/bio_7.tif": "bilinear",
    "Dataset/covariates/raw/bio_8.tif": "bilinear",
    "Dataset/covariates/raw/bio_9.tif": "bilinear",
    "Dataset/covariates/raw/bio_10.tif": "bilinear",
    "Dataset/covariates/raw/bio_11.tif": "bilinear",
    "Dataset/covariates/raw/bio_12.tif": "bilinear",
    "Dataset/covariates/raw/bio_13.tif": "bilinear",
    "Dataset/covariates/raw/bio_14.tif": "bilinear",
    "Dataset/covariates/raw/bio_15.tif": "bilinear",
    "Dataset/covariates/raw/bio_16.tif": "bilinear",
    "Dataset/covariates/raw/bio_17.tif": "bilinear",
    "Dataset/covariates/raw/bio_18.tif": "bilinear",
    "Dataset/covariates/raw/bio_19.tif": "bilinear",
}

covariate_arrays = []

for raster_path, method_name in resampling_by_raster.items():
    method = raster_utils.get_resampling(method_name)
    out_path = os.path.join(
        "Dataset/covariates/downsampled", f"{Path(raster_path).stem}_27km2.tif"
    )

    print(f"Resampling {raster_path} using {method_name}")

    arr = raster_utils.resample_raster_to_grid(
        src_path=raster_path,
        out_path=out_path,
        target_crs=target_crs,
        transform=transform,
        width=width,
        height=height,
        resampling_method=method,
    )
    covariate_arrays.append(arr)

######

# Step 4 : Build valid domain mask from AOI and covariate rasters

######

domain_mask = raster_utils.build_domain_mask_from_aoi_and_covariates(
    aoi_mask, covariate_arrays
)

######

# Step 5 : Build raster Q as described in Merow et al. (2016)

###

PIN = 0.8
R_DECAY = 0.05
S_SHAPE = 1.0
K_SHIFT = 0.0
Q_EPS = 1e-9

signed_dist = baseline_merow.signed_distance_km(expert_mask, transform)

Q = baseline_merow.build_offset_Q(
    expert_mask=expert_mask,
    domain_mask=domain_mask,
    signed_dist_km=signed_dist,
    pin=PIN,
    r=R_DECAY,
    s=S_SHAPE,
    k=K_SHIFT,
)

Q = baseline_merow.sanitize_and_normalize_Q(Q, domain_mask, q_eps=Q_EPS)

pin_str = str(PIN).replace(".", "")
rdecay_str = str(R_DECAY).replace(".", "")
sshape_str = str(S_SHAPE).replace(".", "")
kshift_str = str(K_SHIFT).replace(".", "")
q_path = os.path.join(
    output_dir,
    f"rasters_q/{species}/pin{pin_str}_r{rdecay_str}_s{sshape_str}_k{kshift_str}.tif",
)


raster_utils.save_array_as_raster(
    path=q_path,
    arr=Q.astype(np.float32),
    transform=transform,
    crs=target_crs,
    nodata=np.nan,
    dtype="float32",
)

print(f"Saved Q raster to: {q_path}")

######

# Step 6 : Project observation data to grid, and remove duplicated presences in each cell

######

obs_csv = f"Dataset/observational_data/full/{species}.csv"
lon_col = "LONGITUDE"
lat_col = "LATITUDE"

obs_gdf = raster_utils.load_and_project_occurrences(
    obs_csv, lon_col, lat_col, target_crs
)
obs_cells = raster_utils.points_to_cells(obs_gdf, transform, width, height)
obs_unique = raster_utils.deduplicate_presences_by_cell(obs_cells, domain_mask)

print(f"Original observations: {len(obs_gdf)}")
print(f"Observations on grid: {len(obs_cells)}")
print(f"Unique occupied cells: {len(obs_unique)}")

occ_x, occ_y = xy(
    transform, obs_unique["row"].values, obs_unique["col"].values, offset="center"
)
occ_cells_gdf = gpd.GeoDataFrame(
    obs_unique.copy(), geometry=gpd.points_from_xy(occ_x, occ_y), crs=target_crs
)

######

# Step 7 : Sample background points

######

N_BACKGROUND = 2000
RANDOM_SEED = 42

bg_cells = raster_utils.sample_background_cells(
    domain_mask=domain_mask,
    presence_cells=obs_unique,
    n_background=N_BACKGROUND,
    seed=RANDOM_SEED,
)

print(f"Background cells sampled: {len(bg_cells)}")


######

# Step 8 : Build training dataset as described in Merow et al. (2016)

######

BACKGROUND_WEIGHT = 0.01


train_df = baseline_merow.build_presence_background_table_merow(
    covariate_arrays=covariate_arrays,
    Q=Q,
    presence_cells=obs_unique,
    bg_cells=bg_cells,
    eps=Q_EPS,
)

train_df = baseline_merow.add_poisson_case_weights(
    train_df, background_weight=BACKGROUND_WEIGHT
)

print("Training table shape:", train_df.shape)
print("Class counts:")
print(train_df["y"].value_counts(dropna=False))
print("logQ min/max:", train_df["logQ"].min(), train_df["logQ"].max())

feature_cols_all = [c for c in train_df.columns if c.startswith("cov_")]

######

# Step 9 : Fit model

######

POISSON_ALPHA = 1.0
Z_CLIP = 5.0
LOGQ_CLIP_MIN = -20.0

model_result, feature_cols, cov_means, cov_stds = baseline_merow.fit_merow_eq1_poisson(
    train_df=train_df,
    weight_col="case_weight",
    alpha=POISSON_ALPHA,
    logQ_clip_min=LOGQ_CLIP_MIN,
    z_clip=Z_CLIP,
)


raw_score, ror = baseline_merow.predict_merow_ror_full_grid_stable(
    covariate_arrays=covariate_arrays,
    domain_mask=domain_mask,
    Q=Q,
    model=model_result,
    feature_cols=feature_cols,
    cov_means=cov_means,
    cov_stds=cov_stds,
    logQ_clip_min=LOGQ_CLIP_MIN,
    z_clip=Z_CLIP,
    eps=Q_EPS,
)


ror_path = os.path.join(output_dir, f"rasters_preds/probs/{species}_merow_27km2.tif")

raster_utils.save_array_as_raster(
    path=ror_path,
    arr=ror,
    transform=transform,
    crs=target_crs,
    nodata=np.nan,
    dtype="float32",
)

print(f"Saved Merow ROR raster to: {ror_path}")


######

# Step 9 : Generate binary range map from ROR raster using percentile method

######

PRESENCE_QUANTILE = 0.10

binary_range, thr = raster_utils.threshold_by_presence_quantile(
    prediction=ror,
    presence_cells=obs_unique,
    domain_mask=domain_mask,
    q=PRESENCE_QUANTILE,
)

print(f"Presence-quantile threshold ({PRESENCE_QUANTILE:.0%}): {thr}")
print(f"Predicted range cells: {np.nansum(binary_range == 1)}")
print(f"Predicted range area (km²): {np.nansum(binary_range == 1) * cell_area_km2}")

binary_save = np.full(binary_range.shape, 255, dtype=np.uint8)
binary_save[np.isfinite(binary_range) & (binary_range == 0)] = 255
binary_save[np.isfinite(binary_range) & (binary_range == 1)] = 1

binary_path = os.path.join(
    output_dir,
    f"rasters_preds/binary/{species}_merow_binary_range_q{int(PRESENCE_QUANTILE*100):02d}_27km2.tif",
)

raster_utils.save_array_as_raster(
    path=binary_path,
    arr=binary_save,
    transform=transform,
    crs=target_crs,
    nodata=255,
    dtype="uint8",
)

print(f"Saved binary range map to: {binary_path}")
