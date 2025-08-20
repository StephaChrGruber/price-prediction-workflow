import logging
from typing import List, Optional
from datetime import datetime

import numpy as np
import polars as pl
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

from constants import TIME_COL, SYMBOL_COL, EPS
from diagnostics import safe_log
import pandas as pd

logger = logging.getLogger(__name__)


def to_dt(s: pl.Series) -> pl.Series:
    return s.dt.replace_time_zone(None)


def prep_stocks(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("Preparing stocks")
    if df.is_empty():
        return df
    d = df.clone()
    needed = ["date", "symbol", "open", "close", "high", "low", "volume"]
    for k in needed:
        if k not in d.columns:
            d = d.with_columns(pl.lit(np.nan).alias(k))
    d = d.rename({"date": TIME_COL, "symbol": SYMBOL_COL})
    d = d.with_columns([
        to_dt(d.get_column(TIME_COL)),
        pl.col("open").cast(pl.Float32, strict=False),
        pl.col("close").cast(pl.Float32, strict=False),
        pl.col("high").cast(pl.Float32, strict=False),
        pl.col("low").cast(pl.Float32, strict=False),
        pl.col("volume").cast(pl.Float32, strict=False),
    ])
    d = d.drop_nulls([TIME_COL, SYMBOL_COL]).sort([SYMBOL_COL, TIME_COL])
    d = d.with_columns(pl.lit("stock").alias("source"))
    out = d.select([SYMBOL_COL, TIME_COL, "open", "close", "high", "low", "volume", "source"])
    logger.info(f"Prepared stocks: shape={out.shape}")
    return out


def prep_crypto(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("Preparing crypto")
    if df.is_empty():
        return df
    d = df.clone()
    needed = [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "num_trades",
        "taker_buy_base_vol",
        "taker_buy_quote_vol",
        "closePctChange",
        "openPctChange",
    ]
    for k in needed:
        if k not in d.columns:
            d = d.with_columns(pl.lit(np.nan).alias(k))
    d = d.rename({"date": TIME_COL, "symbol": SYMBOL_COL})
    d = d.with_columns([to_dt(pl.col(TIME_COL))])
    num_cols = [c for c in needed if c not in {"date", "symbol"}]
    d = d.with_columns([pl.col(c).cast(pl.Float32, strict=False) for c in num_cols])
    d = d.drop_nulls([TIME_COL, SYMBOL_COL]).sort([SYMBOL_COL, TIME_COL])
    d = d.with_columns(pl.lit("crypto").alias("source"))
    out = d.select(
        [
            SYMBOL_COL,
            TIME_COL,
            "open",
            "close",
            "high",
            "low",
            "volume",
            "quote_asset_volume",
            "num_trades",
            "taker_buy_base_vol",
            "taker_buy_quote_vol",
            "closePctChange",
            "openPctChange",
            "source",
        ]
    )
    logger.info(f"Prepared crypto: shape={out.shape}")
    return out

def prep_fx(df: pl.DataFrame) -> pl.DataFrame:
    logger.info("Preparing FX")
    if df.is_empty():
        return df
    d = df.clone()
    d = d.with_columns(d.get_column(TIME_COL))
    d = d.drop_nulls([TIME_COL]).sort(TIME_COL)
    rate_cols = [c for c in d.columns if c not in {TIME_COL, "base_currency"}]
    if not rate_cols:
        return pl.DataFrame({TIME_COL: d[TIME_COL], "eur_fx_logret": np.zeros(len(d), dtype=float)})
    rates = d.select(rate_cols).with_columns([pl.col(c).cast(pl.Float32, strict=False) for c in rate_cols])
    rates = rates.fill_null(strategy="forward").fill_null(strategy="backward").fill_null(1.0)
    rates = rates.select([pl.col(c).clip(lower_bound=EPS) for c in rate_cols])
    log_rates = np.log(rates.to_numpy())
    dlog = np.diff(log_rates, axis=0)
    dlog = np.vstack([np.zeros((1, dlog.shape[1]), dtype=float), dlog])
    eur_fx_logret = np.nanmean(dlog, axis=1)
    eur_fx_logret = np.where(np.isfinite(eur_fx_logret), eur_fx_logret, 0.0).astype(np.float32)
    out = pl.DataFrame({TIME_COL: d[TIME_COL], "eur_fx_logret": eur_fx_logret})
    logger.info(f"Prepared FX: shape={out.shape}")
    return out


def _extract_sentiment(val) -> float:
    if val is None:
        return 0.0
    try:
        arr = np.array(val, dtype=float).flatten()
        if arr.size == 0 or not np.isfinite(arr).any():
            return 0.0
        return float(np.nanmean(arr))
    except Exception:
        return 0.0


def prep_news_global(df: pl.DataFrame, pca_cap: int = 32) -> pl.DataFrame:
    logger.info("Preparing News")
    if df.is_empty():
        return df
    d = df.clone()
    d = d.with_columns(to_dt(d.get_column(TIME_COL)).dt.truncate("1d"))
    d = d.drop_nulls([TIME_COL])
    if "sentiment" in d.columns:
        d = d.with_columns(pl.col("sentiment").map_elements(_extract_sentiment).alias("sentiment_scalar"))
    else:
        d = d.with_columns(pl.lit(0.0).alias("sentiment_scalar"))

    def _to_vec(x):
        try:
            a = np.asarray(x, dtype=float).reshape(-1)
            if a.size == 0 or not np.isfinite(a).any():
                return np.empty((0,), dtype=float)
            return a
        except Exception:
            return np.empty((0,), dtype=float)

    if "embeddings" in d.columns:
        d = d.with_columns(pl.col("embeddings").map_elements(_to_vec).alias("emb"))
    else:
        d = d.with_columns(pl.lit(np.empty((0,), dtype=float)).alias("emb"))

    max_dim = max(d["emb"].map_elements(len).max(), 0)
    dim = min(max_dim, pca_cap)
    if dim:
        emb_mat = np.vstack([v[:dim] for v in d["emb"].to_list()])
        pca = IncrementalPCA(n_components=dim)
        for i in range(0, emb_mat.shape[0], 1000):
            pca.partial_fit(emb_mat[i : i + 1000])
        Z = pca.transform(emb_mat)
        for i in range(dim):
            d = d.with_columns(pl.Series(f"news_pca_{i}", Z[:, i].astype(np.float32)))

    agg_expr = [
        pl.col("sentiment_scalar").mean().alias("news_sent_mean"),
        pl.col("sentiment_scalar").max().alias("news_sent_max"),
        pl.col("sentiment_scalar").count().alias("news_count"),
    ]
    agg_expr.extend(
        [pl.col(f"news_pca_{i}").mean().alias(f"news_pca_{i}") for i in range(dim)]
    )
    out = d.group_by(TIME_COL).agg(agg_expr).sort(TIME_COL)
    out = out.with_columns(pl.all().exclude(TIME_COL).cast(pl.Float32, strict=False))
    logger.info(f"Prepared News: shape={out.shape}")
    return out


def prep_weather_daily(weather_df: pl.DataFrame, agg: str = "mean") -> pl.DataFrame:
    logger.info("Preparing Weather")
    if weather_df.is_empty():
        return pl.DataFrame({TIME_COL: []})
    w = weather_df.clone()
    w = w.with_columns(to_dt(pl.col(TIME_COL)))
    w = w.drop_nulls([TIME_COL])
    wx_num = [c for c in ["tavg", "tmin", "tmax", "prcp", "snow", "wdir", "wspd", "wpgt", "pres", "tsun"] if c in w.columns]
    w = w.with_columns([pl.col(c).cast(pl.Float32, strict=False) for c in wx_num])
    if agg == "median":
        wagg = w.group_by(TIME_COL).median()
    else:
        wagg = w.group_by(TIME_COL).mean()
    wagg = wagg.select([TIME_COL, *wx_num]).sort(TIME_COL)
    wagg = wagg.rename({c: f"wx_{c}" for c in wx_num})
    logger.info(f"Prepared Weather: shape={wagg.shape}")
    return wagg


def fit_weather_pca(weather_daily: pl.DataFrame, n_components: int, fit_dates: Optional[List[datetime]] = None):
    wx_cols = [c for c in weather_daily.columns if c.startswith("wx_")]
    if not wx_cols:
        logger.info("No weather columns for PCA")
        return None, None, [], []
    df = weather_daily.clone()
    if fit_dates is not None:
        df = df.filter(pl.col(TIME_COL).is_in(fit_dates))
        if df.is_empty():
            logger.info("No weather rows for given fit dates")
            return None, None, wx_cols, []
    sc = StandardScaler()
    Xfit = sc.fit_transform(df.select(wx_cols).to_numpy()).astype(np.float32)
    pca = IncrementalPCA(n_components=min(n_components, Xfit.shape[1]))
    for i in range(0, Xfit.shape[0], 20000):
        pca.partial_fit(Xfit[i : i + 20000])
    names = [f"wx_pca_{i}" for i in range(pca.n_components_)]
    logger.info(f"Fitted weather PCA: components={len(names)}")
    return pca, sc, wx_cols, names


def transform_weather_pca(weather_daily: pl.DataFrame, pca, sc, wx_cols: List[str], names: List[str]):
    if pca is None or not wx_cols:
        out = weather_daily.select([TIME_COL])
        for n in names:
            out = out.with_columns(pl.lit(0.0).alias(n))
        logger.info("Weather PCA transform skipped, returning zeros")
        return out
    X = sc.transform(weather_daily.select(wx_cols).to_numpy()).astype(np.float32)
    Z = pca.transform(X).astype(np.float32)
    out = weather_daily.select([TIME_COL]).clone()
    for i, n in enumerate(names):
        out = out.with_columns(pl.Series(n, Z[:, i].astype(np.float32)))
    logger.info(f"Transformed weather PCA: shape={out.shape}")
    return out


def merge_weather(panel: pd.DataFrame, weather_daily: pd.DataFrame, n_components: int, fit_dates: Optional[List[datetime]] = None):
    weather_daily = weather_daily.fill_null(0.0)
    if weather_daily.is_empty():
        names = [f"wx_pca_{i}" for i in range(n_components)]
        out = panel.clone()
        for n in names:
            out = out.with_columns(pl.lit(0.0).alias(n))
        logger.info("Weather data empty, filled zeros")
        return out, None, None, names
    pca, sc, wx_cols, names = fit_weather_pca(weather_daily, n_components, fit_dates)
    wxz = transform_weather_pca(weather_daily, pca, sc, wx_cols, names)
    wxz_pd = wxz.to_pandas() if hasattr(wxz, "to_pandas") else wxz
    out = panel.merge(wxz_pd, on=TIME_COL, how="left")
    out = pl.from_pandas(out)
    for n in names:
        out = out.with_columns(pl.col(n).fill_null(0.0))
    logger.info(f"Merged weather data: shape={out.shape}")
    return out.to_pandas(), pca, sc, names
