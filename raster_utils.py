import math
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin, rowcol
from rasterio.warp import reproject, Resampling


def build_domain_mask_from_aoi_and_covariates(aoi_mask, covariate_arrays):
    mask = aoi_mask.copy().astype(bool)
    for arr in covariate_arrays:
        mask &= np.isfinite(arr)
    return mask


def build_target_grid_from_aoi(aoi_shp, target_crs, cell_size_m):
    """
    Build a grid from AOI bounds.
    Each grid cell will have width = cell_size_m

    Parameters:
        - aoi_shp : Shapefile of the AOI area
        - target_crs : Target CRS (UTM CRS is recommended)
        - cell_size_m : Grid cell width (in meters)

    Output :
        - aoi : AOI reprojected into the target CRS
        - transform : Affine transform to allow mapping grid cell coordinates (pixel) to real-world (e.g. northing-easting for UTM CRS) coordinate
        - width : width of the AOI (in number of cells)
        - height : height of the AOI (in number of cells)
    """
    aoi = gpd.read_file(aoi_shp).to_crs(target_crs)
    minx, miny, maxx, maxy = aoi.total_bounds

    width = int(math.ceil((maxx - minx) / cell_size_m))
    height = int(math.ceil((maxy - miny) / cell_size_m))

    adjusted_maxy = miny + height * cell_size_m
    transform = from_origin(minx, adjusted_maxy, cell_size_m, cell_size_m)

    return aoi, transform, width, height


def rasterize_polygon_to_grid(gdf, transform, width, height):
    """
    Rasterizes polygon(s) into a pre-defined grid through the use of an affine transform.

    Input :
        - gdf : Geodataframe containing the polygon
        - transform : Affine transform associated with the desired grid
        - width : width of the grid (in number of cells)
        - height : height of the grid (in number of cells)

    Output :
        - arr : Binary raster in which each grid cell that were inside the polygon(s) will have a value of 1 and 0 otherwise
    """
    shapes = [
        (geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty
    ]
    arr = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    )
    return arr


def get_resampling(method_name):
    """
    Returns the appropriate rasterio resampling function

    Input:
        - method_name : name of the resampling function (string)
    Output:
        - Rasterio Resampling function
    """
    if method_name == "nearest":
        return Resampling.nearest
    if method_name == "bilinear":
        return Resampling.bilinear
    if method_name == "cubic":
        return Resampling.cubic
    if method_name == "average":
        return Resampling.average
    raise ValueError(f"Unsupported resampling method: {method_name}")


def resample_raster_to_grid(
    src_path, out_path, target_crs, transform, width, height, resampling_method
):
    """
    Reproject and resample a raster in order for it to match a target grid.

    Input:
        - src_path : Path of the input raster file (str)
        - out_path : Path where the resampled raster will be saved (str)
        - target_crs : Target coordinate reference system (UTM recommended)
        - transform : Affine transform defining the target grid
        - width : Width of the target grid (in number of cells)
        - height : Height of the target grid (in number of cells)
        - resampling_method : Resampling function to use (e.g. bilinear)

    Returns :
        - dst_arr : np.ndarray
            2D array (height x width) containing the resampled raster values,
            aligned to the target grid. Missing values are set to NaN.
    """

    with rasterio.open(src_path) as src:
        src_nodata = src.nodata
        dst_arr = np.full((height, width), np.nan, dtype=np.float32)

        reproject(
            source=rasterio.band(src, 1),
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=transform,
            dst_crs=target_crs,
            dst_nodata=np.nan,
            resampling=resampling_method,
        )

        save_array_as_raster(
            path=out_path,
            arr=dst_arr,
            transform=transform,
            crs=target_crs,
            nodata=np.nan,
            dtype="float32",
        )

    return dst_arr


def save_array_as_raster(path, arr, transform, crs, nodata, dtype):
    """
    Saves a numpy array to a raster (.tif) file.

    Input:
        - path : Output file path (should end with .tif)
        - arr : np.ndarray containing raster values to save
        - transform :  Affine transform defining the spatial reference of the raster
        - crs : Coordinate reference system of the raster (UTM recommended)
        - nodata : Value used to represent missing or invalid data in the raster
        - dtype : Data type of the output raster (e.g. "float32" or "uint8")

    Returns:
        None (saves the raster to the specified path)
    """
    meta = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(arr.astype(dtype), 1)


def load_and_project_occurrences(obs_csv, lon_col, lat_col, target_crs):
    df = pd.read_csv(obs_csv)
    df = df[[lon_col, lat_col]].dropna().copy()

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326"
    ).to_crs(target_crs)

    return gdf


def points_to_cells(gdf, transform, width, height):
    rows = []
    cols = []

    for geom in gdf.geometry:
        r, c = rowcol(transform, geom.x, geom.y)
        rows.append(r)
        cols.append(c)

    out = gdf.copy()
    out["row"] = rows
    out["col"] = cols

    out = out[
        (out["row"] >= 0)
        & (out["row"] < height)
        & (out["col"] >= 0)
        & (out["col"] < width)
    ].copy()

    return out


def deduplicate_presences_by_cell(obs_cells, domain_mask):
    obs_cells["valid_domain"] = domain_mask[
        obs_cells["row"].values, obs_cells["col"].values
    ]
    obs_cells = obs_cells[obs_cells["valid_domain"]].copy()
    obs_unique = obs_cells.drop_duplicates(subset=["row", "col"]).copy()
    return obs_unique


def sample_background_cells(domain_mask, presence_cells, n_background, seed=42):
    rng = np.random.default_rng(seed)

    valid_rows, valid_cols = np.where(domain_mask)
    all_cells = pd.DataFrame({"row": valid_rows, "col": valid_cols})

    presence_set = set(zip(presence_cells["row"], presence_cells["col"]))
    keep = ~all_cells.apply(lambda x: (x["row"], x["col"]) in presence_set, axis=1)
    bg_candidates = all_cells[keep].copy()

    n_take = min(n_background, len(bg_candidates))
    idx = rng.choice(bg_candidates.index.values, size=n_take, replace=False)
    bg = bg_candidates.loc[idx].reset_index(drop=True)

    return bg

def threshold_by_presence_quantile(prediction, presence_cells, domain_mask, q=0.10):
    rr = presence_cells["row"].values
    cc = presence_cells["col"].values

    vals = prediction[rr, cc]
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        raise ValueError("No finite prediction values found at presence cells.")

    thr = float(np.quantile(vals, q))

    binary = np.full(prediction.shape, np.nan, dtype=np.float32)
    binary[domain_mask] = 0.0
    binary[domain_mask & np.isfinite(prediction) & (prediction >= thr)] = 1.0

    return binary, thr


def extract_covariates_from_cells(covariate_arrays, cell_df):
    rr = cell_df["row"].values
    cc = cell_df["col"].values

    data = {}
    for i, arr in enumerate(covariate_arrays, start=1):
        data[f"cov_{i}"] = arr[rr, cc]

    return pd.DataFrame(data)
