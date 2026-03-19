import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
import statsmodels.api as sm
from raster_utils import extract_covariates_from_cells
from sklearn.linear_model import PoissonRegressor


def signed_distance_km(expert_mask, transform):
    """
    Positive inside, negative outside, in km.
    """
    xres = abs(transform.a)
    yres = abs(transform.e)
    pixel_size_km = ((xres + yres) / 2.0) / 1000.0

    inside_dist = distance_transform_edt(expert_mask) * pixel_size_km
    outside_dist = distance_transform_edt(~expert_mask) * pixel_size_km

    signed = inside_dist.copy()
    signed[~expert_mask] = -outside_dist[~expert_mask]
    return signed


def build_offset_Q(
    expert_mask, domain_mask, signed_dist_km, pin=0.8, r=0.05, s=1.0, k=0.0
):
    """
    Build smoothed offset Q(x), normalized across the modeling domain.
    """
    d = signed_dist_km
    h = 1.0 / np.power(1.0 + np.exp(-r * (d - k)), 1.0 / s)

    valid = domain_mask
    inside = expert_mask & valid
    outside = (~expert_mask) & valid

    m = inside.sum()
    n = outside.sum()

    A = h[inside].sum()
    B = h[outside].sum()
    N = m + n
    S = A + B

    numerator = m - pin * N
    denominator = pin * S - A

    t = numerator / denominator

    q_raw = 1.0 + t * h
    q_raw[~valid] = 0.0

    total = q_raw.sum()

    Q = q_raw / total
    return Q


def fit_presence_background_glm(train_df):
    feature_cols = [c for c in train_df.columns if c.startswith("cov_")]

    eps = 1e-12
    train_df = train_df.copy()
    train_df["logQ"] = np.clip(train_df["logQ"], np.log(eps), None)

    X_raw = train_df[feature_cols].copy()
    cov_means = X_raw.mean()
    cov_stds = X_raw.std(ddof=0).replace(0, 1.0)

    X_scaled = (X_raw - cov_means) / cov_stds
    X_scaled = X_scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X = sm.add_constant(X_scaled, has_constant="add")
    y = train_df["y"].values

    logQ_center = train_df["logQ"].mean()
    offset = train_df["logQ"].values - logQ_center

    model = sm.GLM(y, X, family=sm.families.Binomial(), offset=offset)
    result = model.fit(maxiter=200)

    return result, feature_cols, cov_means, cov_stds, logQ_center


def predict_full_grid(
    covariate_arrays,
    domain_mask,
    Q,
    model_result,
    feature_cols,
    cov_means=None,
    cov_stds=None,
    logQ_center=0.0,
):
    H, W = domain_mask.shape
    P = len(covariate_arrays)

    stack = np.stack(covariate_arrays, axis=-1)
    flat = stack.reshape(-1, P)
    valid_flat = domain_mask.reshape(-1)

    X_full = pd.DataFrame(flat[valid_flat], columns=feature_cols)

    if cov_means is not None and cov_stds is not None:
        X_full = (X_full - cov_means) / cov_stds
        X_full = X_full.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X_full = sm.add_constant(X_full, has_constant="add")

    eps = 1e-12
    logQ = np.log(np.clip(Q.reshape(-1)[valid_flat], eps, None)) - logQ_center

    linpred = model_result.predict(X_full, offset=logQ, linear=True)

    linpred = np.asarray(linpred, dtype=np.float64)
    m = np.nanmax(linpred)
    raw = np.exp(linpred - m)
    s = raw.sum()

    raw = raw / s

    pred = np.full(H * W, np.nan, dtype=np.float64)
    pred[valid_flat] = raw
    return pred.reshape(H, W).astype(np.float32)


def sanitize_and_normalize_Q(Q, domain_mask, q_eps=1e-9):
    Q = np.where(domain_mask, Q, np.nan)
    Q = np.where(domain_mask, np.clip(Q, q_eps, None), np.nan)

    total = np.nansum(Q)

    Q = Q / total
    return Q


def extract_Q_from_cells(Q, cell_df):
    rr = cell_df["row"].values
    cc = cell_df["col"].values
    return Q[rr, cc]


def build_presence_background_table_merow(
    covariate_arrays, Q, presence_cells, bg_cells, eps=1e-9
):
    X_pres = extract_covariates_from_cells(covariate_arrays, presence_cells)
    X_bg = extract_covariates_from_cells(covariate_arrays, bg_cells)

    q_pres = np.clip(extract_Q_from_cells(Q, presence_cells), eps, None)
    q_bg = np.clip(extract_Q_from_cells(Q, bg_cells), eps, None)

    pres_df = X_pres.copy()
    pres_df["y"] = 1.0
    pres_df["logQ"] = np.log(q_pres)
    pres_df["row"] = presence_cells["row"].values
    pres_df["col"] = presence_cells["col"].values

    bg_df = X_bg.copy()
    bg_df["y"] = 0.0
    bg_df["logQ"] = np.log(q_bg)
    bg_df["row"] = bg_cells["row"].values
    bg_df["col"] = bg_cells["col"].values

    train_df = pd.concat([pres_df, bg_df], ignore_index=True)
    train_df = (
        train_df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    )
    return train_df


def add_poisson_case_weights(train_df, background_weight=0.01):
    df = train_df.copy()
    df["case_weight"] = np.where(df["y"].values == 1.0, 1.0, background_weight)
    return df


def fit_merow_eq1_poisson(
    train_df, weight_col="case_weight", alpha=1.0, logQ_clip_min=-20.0, z_clip=5.0
):
    df = train_df.copy()

    feature_cols = [c for c in df.columns if c.startswith("cov_")]
    keep_cols = feature_cols + ["y", "logQ"]
    if weight_col is not None and weight_col in df.columns:
        keep_cols.append(weight_col)

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=keep_cols).copy()

    y = df["y"].astype(float).values

    df["logQ"] = df["logQ"].astype(float).clip(lower=logQ_clip_min)
    logQ = df["logQ"].values
    Q_local = np.exp(logQ)

    X_raw = df[feature_cols].copy()

    std0 = X_raw.std(ddof=0)
    feature_cols = std0[std0 > 1e-8].index.tolist()
    X_raw = X_raw[feature_cols]
    cov_means = X_raw.mean()
    cov_stds = X_raw.std(ddof=0).replace(0, 1.0)

    X = (X_raw - cov_means) / cov_stds
    X = X.clip(lower=-z_clip, upper=z_clip)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    sample_weight = None
    if weight_col is not None and weight_col in df.columns:
        sample_weight = df[weight_col].astype(float).values

    y_star = y / Q_local
    w_star = Q_local if sample_weight is None else sample_weight * Q_local

    model = PoissonRegressor(alpha=alpha, fit_intercept=True, max_iter=1000, tol=1e-8)
    model.fit(X, y_star, sample_weight=w_star)

    return model, feature_cols, cov_means, cov_stds


def predict_merow_ror_full_grid_stable(
    covariate_arrays,
    domain_mask,
    Q,
    model,
    feature_cols,
    cov_means,
    cov_stds,
    logQ_clip_min=-20.0,
    z_clip=5.0,
    eps=1e-12,
):
    H, W = domain_mask.shape
    P = len(covariate_arrays)

    stack = np.stack(covariate_arrays, axis=-1)
    flat = stack.reshape(-1, P)
    valid_flat = domain_mask.reshape(-1)

    X_full = pd.DataFrame(
        flat[valid_flat], columns=[f"cov_{i}" for i in range(1, P + 1)]
    )
    X_full = X_full[feature_cols]

    X_full = (X_full - cov_means) / cov_stds
    X_full = X_full.clip(lower=-z_clip, upper=z_clip)
    X_full = X_full.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    q_valid = np.clip(Q.reshape(-1)[valid_flat], eps, None)
    logQ_valid = np.log(q_valid)
    logQ_valid = np.clip(logQ_valid, logQ_clip_min, None)

    lp = model.intercept_ + X_full.values @ model.coef_
    eta = lp + logQ_valid
    eta = np.clip(eta, -700, 700)

    raw_valid = np.exp(eta)

    total = np.nansum(raw_valid)
    ror_valid = raw_valid / total

    raw_full = np.full(H * W, np.nan, dtype=np.float64)
    ror_full = np.full(H * W, np.nan, dtype=np.float64)

    raw_full[valid_flat] = raw_valid
    ror_full[valid_flat] = ror_valid

    return raw_full.reshape(H, W).astype(np.float32), ror_full.reshape(H, W).astype(
        np.float32
    )
