#!/usr/bin/env python3
"""
train_custom.py — Robust multi‑horizon forecaster (schemas adapted + NaN guards)
============================================================================== 

- Adapts to your Mongo collections with null‑tolerant loaders.
- Uses safe logs, clamps, and NaN/Inf purging to avoid "Train nan | Val nan".
- Predicts 1m / 6m / 1y (trading‑day or calendar) for stocks & crypto.
- Features: technicals, FX (EUR drift), global news (sentiment + PCA[embeddings]),
  global weather (station metrics → PCA), time features.
- Options: quantile outputs (p50/p90), walk‑forward CV, leakage‑safe scaling.
- Persists top‑K predictions to Mongo and saves artifacts.

Run (example):
  python train_custom.py --mongo-uri "..." --db PriceForecast \
    --coll-stock DailyStockData --coll-crypto DailyCryptoData \
    --coll-fx DailyCurrencyExchangeRates --coll-news NewsHeadlines \
    --coll-weather DailyWeatherData --coll-pred Predictions

"""

from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from datetime import datetime, timedelta
import asyncio

import argparse
import copy
import logging
import os
import sys
import gc
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import joblib

from data_utils import (
    mongo_client,
    iter_mongo_df_chunks,
    stage_collection_with_schema,
    stage_collection,
    align_df_to_schema,
    sanitize_list_column,
    table_from_df_and_schema,
)
from diagnostics import setup_diagnostics, log_mem, set_seed, safe_log
from preprocessing import (
    to_dt,
    prep_stocks,
    prep_crypto,
    prep_fx,
    prep_news_global,
    prep_weather_daily,
    merge_weather,
    fit_weather_pca,
    transform_weather_pca,
)
from constants import TIME_COL, SYMBOL_COL, EPS
from memory_utils import reduce_mem_usage

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)-15s] %(asctime)s [%(levelname)-8s]: %(message)s",
    stream=sys.stdout,
    force=True
)

logger = logging.getLogger(__name__)

setup_diagnostics()
logger.info(f"[boot] torch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")

# ==========================
# Targets
# ==========================

def add_calendar_targets(panel: pd.DataFrame, horizons_days: List[int]):
    logger.info("Adding calendar targets")
    panel = panel.sort_values([SYMBOL_COL, TIME_COL]).copy()
    tcols, mcols = [], []
    for h in horizons_days:
        logger.info(f"Processing horizon {h}")
        tcol, mcol = f"target_{h}d", f"mask_{h}d"
        panel[tcol] = panel.groupby(SYMBOL_COL)["close"].transform(lambda s: (safe_log(s.shift(-h)) - safe_log(s)))
        panel[mcol] = panel[tcol].replace([np.inf,-np.inf], np.nan).notna().astype(int)
        tcols.append(tcol); mcols.append(mcol)
    logger.info(f"Added calendar targets: cols={tcols}")
    panel = reduce_mem_usage(panel, sparse_threshold=0.9)
    return panel, tcols, mcols


def add_trading_targets(panel: pd.DataFrame, horizons_td: List[int]):
    logger.info("Adding trading-day targets")
    pieces = []
    for sym, g in panel.groupby(SYMBOL_COL, sort=False):
        g = g.sort_values(TIME_COL).copy()
        td_idx = g.index[g["is_trading_day"]==1]
        logc = safe_log(g.loc[td_idx, "close"].astype(float).values)
        for h in horizons_td:
            tcol, mcol = f"target_td{h}", f"mask_td{h}"
            fut = np.roll(logc, -h); targ = fut - logc; targ[-h:] = np.nan
            g.loc[td_idx, tcol] = targ; g.loc[td_idx, mcol] = (~np.isnan(targ)).astype(int)
            g.loc[g["is_trading_day"]==0, tcol] = np.nan; g.loc[g["is_trading_day"]==0, mcol] = 0
        pieces.append(g)
    panel2 = pd.concat(pieces).sort_values([SYMBOL_COL, TIME_COL])
    tcols = [f"target_td{h}" for h in horizons_td]; mcols = [f"mask_td{h}" for h in horizons_td]
    logger.info(f"Added trading targets: cols={tcols}")
    panel2 = reduce_mem_usage(panel2, sparse_threshold=0.9)
    return panel2, tcols, mcols

# ==========================
# Sequences / scaling / splits
# ==========================

def build_sequences_multi(
    panel: pd.DataFrame,
    lookback: int,
    features: List[str],
    target_cols: List[str],
    mask_cols: List[str],
    require_all: bool = True,
):
    logger.info("Building sequences")
    panel = panel.sort_values([SYMBOL_COL, TIME_COL])
    X, M, A, Y, D = [], [], [], [], []
    H = len(target_cols)

    for sym, g in panel.groupby(SYMBOL_COL, observed=False):
        logger.info(f"Building sequence for [{sym}]")
        g = g.reset_index(drop=True)

        # Convert to NumPy arrays once per symbol to minimise memory usage
        feat_arr = g[features].to_numpy(np.float32, copy=True)
        mask_trading = g["is_trading_day"].to_numpy(bool, copy=True)
        asset_ids = g["asset_id"].to_numpy(np.int64, copy=True)
        targets_arr = g[target_cols].to_numpy(np.float32, copy=True)
        masks_arr = g[mask_cols].to_numpy(int, copy=True)
        dates_arr = pd.to_datetime(g[TIME_COL]).to_numpy("datetime64[ns]")

        for t in range(lookback, len(g)):
            y = targets_arr[t]
            masks = masks_arr[t]
            if require_all:
                if not np.isfinite(y).all() or masks.sum() < H:
                    continue
            else:
                if not np.isfinite(y).any():
                    continue

            win_slice = slice(t - lookback, t)
            win_mask = mask_trading[win_slice]
            if not win_mask.any():
                continue

            X.append(feat_arr[win_slice])
            M.append(win_mask)
            A.append(int(asset_ids[t - 1]))
            Y.append(y)
            D.append(dates_arr[t])

    X = np.stack(X) if X else np.zeros((0, lookback, len(features)), np.float32)
    M = np.stack(M) if M else np.zeros((0, lookback), bool)
    A = np.array(A, np.int64)
    Y = np.stack(Y) if Y else np.zeros((0, H), np.float32)
    D = np.array(D, dtype="datetime64[ns]")
    logger.info(f"Built sequences: X={X.shape} Y={Y.shape}")
    return X, M, A, Y, D


def time_split_idx(n: int, val_ratio: float = 0.2):
    logger.info(f"Creating time split n={n} val_ratio={val_ratio}")
    val_n = int(n * val_ratio)
    idx = np.arange(n)
    tr, va = idx[: n - val_n], idx[n - val_n :]
    logger.info(f"Split sizes: train={len(tr)} val={len(va)}")
    return tr, va


def walk_forward_splits(dates,
                        train_span_days=365*3,
                        val_span_days=90,
                        step_days=90):
    logger.info(
        f"Generating walk-forward splits train_span={train_span_days} val_span={val_span_days} step={step_days}"
    )
    # Normalize to numpy datetime64[ns] array (keeps order, no index semantics)
    dates = pd.to_datetime(dates)
    dates = dates.to_numpy() if hasattr(dates, "to_numpy") else np.asarray(dates, dtype="datetime64[ns]")

    if dates.size == 0:
        return  # nothing to yield

    start = dates.min()
    end = dates.max()

    train_span = np.timedelta64(train_span_days, "D")
    val_span = np.timedelta64(val_span_days, "D")
    step = np.timedelta64(step_days, "D")

    anchor = start + train_span
    i = 0
    while True:
        train_end = anchor
        val_end = train_end + val_span
        if train_end > end or val_end > end:
            break

        # Boolean masks are NumPy arrays already — no .to_numpy() here
        tr_mask = (dates > (train_end - train_span)) & (dates <= train_end)
        va_mask = (dates > train_end) & (dates <= val_end)

        tr_idx = np.flatnonzero(tr_mask)
        va_idx = np.flatnonzero(va_mask)

        if tr_idx.size and va_idx.size:
            logger.info(
                f"[split {i}] train={tr_idx.size} val={va_idx.size} anchor={anchor}"
            )
            yield tr_idx, va_idx
            i += 1

        anchor = anchor + step


def fit_scaler_on_train(Xtr: np.ndarray) -> StandardScaler:
    if Xtr.size == 0:
        raise RuntimeError("Empty training set after filtering.")
    logger.info(f"Fitting scaler on data shape={Xtr.shape}")
    N, T, F = Xtr.shape
    sc = StandardScaler().fit(Xtr.reshape(N * T, F))
    logger.info("Scaler fitted")
    return sc


def apply_scaler(X: np.ndarray, sc: StandardScaler) -> np.ndarray:
    if X.size == 0:
        logger.info("apply_scaler received empty array")
        return X
    logger.debug(f"Applying scaler to array shape={X.shape}")
    N, T, F = X.shape
    return sc.transform(X.reshape(N * T, F)).reshape(N, T, F)

# ==========================
# Models & training
# ==========================
class AttentionPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        # x: [B, T, D], mask: [B, T] bool
        scores = self.attn(x).squeeze(-1)           # [B, T]

        if mask is None:
            w = torch.softmax(scores, dim=1)        # [B, T]
        else:
            mask = mask.bool()
            # Large negative for masked positions; will zero after softmax
            scores = scores.masked_fill(~mask, -1e9)
            w = torch.softmax(scores, dim=1)

            # Zero weights where masked, then renormalize
            w = w * mask.float()
            denom = w.sum(dim=1, keepdim=True).clamp_min(1e-8)

            # If a row is fully masked, fallback to uniform over all timesteps
            all_false = (~mask).all(dim=1)
            if all_false.any():
                w[all_false] = 1.0 / x.size(1)

            w = w / denom

        pooled = (x * w.unsqueeze(-1)).sum(dim=1)   # [B, D]
        return pooled, w


class PriceForecastMulti(nn.Module):
    def __init__(self, n_features, n_assets, out_dim, hidden=128, layers=2, dropout=0.15, emb_dim=32):
        super().__init__()
        self.asset_emb = nn.Embedding(n_assets, emb_dim)
        self.proj = nn.Linear(n_features + emb_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.attn = AttentionPool(hidden)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden//2, out_dim))
    def forward(self, x, asset_id, mask=None):
        emb = self.asset_emb(asset_id).unsqueeze(1).expand(-1, x.size(1), -1)
        z = self.proj(torch.cat([x, emb], -1))
        out, _ = self.lstm(z); pooled, _ = self.attn(out, mask)
        return self.head(pooled)

class PriceForecastQuantilesMulti(nn.Module):
    def __init__(self, n_features, n_assets, H, quantiles=(0.5,0.9), hidden=128, layers=2, dropout=0.15, emb_dim=32):
        super().__init__(); self.H=H; self.Q=len(quantiles); self.quantiles=list(quantiles)
        self.asset_emb = nn.Embedding(n_assets, emb_dim)
        self.proj = nn.Linear(n_features + emb_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.attn = AttentionPool(hidden)
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden//2, self.H*self.Q))
    def forward(self, x, asset_id, mask=None):
        emb = self.asset_emb(asset_id).unsqueeze(1).expand(-1, x.size(1), -1)
        z = self.proj(torch.cat([x, emb], -1))
        out, _ = self.lstm(z); pooled, _ = self.attn(out, mask)
        return self.out(pooled).view(-1, self.H, self.Q)

def pinball_loss_multi(pred: torch.Tensor, target: torch.Tensor, quantiles: List[float]):
    B,H,Q = pred.shape; t = target.unsqueeze(-1).expand(B,H,Q); e = t - pred
    losses = [torch.maximum(q*e[:,:,i], (q-1)*e[:,:,i]).mean() for i,q in enumerate(quantiles)]
    return torch.stack(losses).mean()

class SeqDS(Dataset):
    def __init__(self, X, M, A, Y): self.X=torch.from_numpy(X); self.M=torch.from_numpy(M); self.A=torch.from_numpy(A); self.Y=torch.from_numpy(Y)
    def __len__(self): return len(self.Y)
    def __getitem__(self,i): return self.X[i], self.M[i], self.A[i], self.Y[i]


def _first_batch_nan_guard(x, y):
    if torch.isnan(x).any() or torch.isinf(x).any():
        logger.info("[FATAL] Non-finite in X batch"); idx = torch.where(~torch.isfinite(x)); logger.info(f"indices: {[t[:5].tolist() for t in idx]}"); raise SystemExit(1)
    if torch.isnan(y).any() or torch.isinf(y).any():
        logger.info("[FATAL] Non-finite in Y batch"); idx = torch.where(~torch.isfinite(y)); logger.info(f"indices: {[t[:5].tolist() for t in idx]}"); raise SystemExit(1)


def train_point(model, tr_dl, va_dl, device="cpu", epochs=10, lr=7e-4, weight_decay=1e-4, patience=15):
    logger.info("Starting point training")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss(); best=float('inf'); best_state=copy.deepcopy(model.state_dict()); bad=0
    model.to(device)
    for ep in range(1, epochs+1):
        model.train(); tl=0.0; first=True
        for x,m,a,y in tqdm(tr_dl, desc=f"Ep {ep}/{epochs}"):
            x,m,a,y = x.to(device), m.to(device), a.to(device), y.to(device)
            if first: _first_batch_nan_guard(x,y); first=False
            opt.zero_grad(); pred = model(x,a,m); loss = loss_fn(pred,y)
            if torch.isnan(loss) or torch.isinf(loss): logger.info("[FATAL] loss non-finite"); raise SystemExit(1)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
            tl += loss.item()*len(y)
        tl /= len(tr_dl.dataset) if len(tr_dl.dataset) else float('nan')
        model.eval(); vl=0.0
        with torch.no_grad():
            for x,m,a,y in va_dl:
                x,m,a,y = x.to(device), m.to(device), a.to(device), y.to(device)
                vloss = nn.functional.smooth_l1_loss(model(x,a,m), y)
                vl += vloss.item()*len(y)
        vl /= len(va_dl.dataset) if len(va_dl.dataset) else float('nan')
        logger.info(f"Train {tl:.5f} | Val {vl:.5f}")
        if vl < best - 1e-4: best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad > patience: logger.info("Early stop")
            break
    model.load_state_dict(best_state)
    logger.info("Finished point training")
    return model, best


def train_quant(model, tr_dl, va_dl, device="cpu", epochs=10, lr=7e-4, weight_decay=1e-4, patience=5):
    logger.info("Starting quantile training")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay); Q = model.quantiles
    best=float('inf'); best_state=copy.deepcopy(model.state_dict()); bad=0
    model.to(device)
    for ep in range(1, epochs+1):
        model.train(); tl=0.0; first=True
        for x,m,a,y in tqdm(tr_dl, desc=f"[Q] Ep {ep}/{epochs}"):
            x,m,a,y = x.to(device), m.to(device), a.to(device), y.to(device)
            if first: _first_batch_nan_guard(x,y); first=False
            opt.zero_grad(); pred = model(x,a,m); loss = pinball_loss_multi(pred,y,Q)
            if torch.isnan(loss) or torch.isinf(loss): logger.info("[FATAL] loss non-finite"); raise SystemExit(1)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
            tl += loss.item()*len(y)
        tl /= len(tr_dl.dataset) if len(tr_dl.dataset) else float('nan')
        model.eval(); vl=0.0
        with torch.no_grad():
            for x,m,a,y in va_dl:
                x,m,a,y = x.to(device), m.to(device), a.to(device), y.to(device)
                vloss = pinball_loss_multi(model(x,a,m), y, Q)
                vl += vloss.item()*len(y)
        vl /= len(va_dl.dataset) if len(va_dl.dataset) else float('nan')
        logger.info(f"[Q] Train {tl:.5f} | Val {vl:.5f}")
        if vl < best - 1e-4: best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad>patience:
                logger.info("Early stop")
                break
    model.load_state_dict(best_state)
    logger.info("Finished quantile training")
    return model, best

# ==========================
# Scoring & persistence
# ==========================

def score_point(panel, FEATURES, scaler, model, HORIZONS, HLAB, device):
    logger.info("Scoring point predictions")
    rows = []
    L = args.lookback
    for sym, g in panel.groupby(SYMBOL_COL):
        g = g.sort_values(TIME_COL)
        if len(g) < L: continue
        win = g.iloc[-L:]
        x = win[FEATURES].values.astype(np.float32)[None,:,:]; x = apply_scaler(x, scaler)
        m = win["is_trading_day"].values.astype(bool)[None,:]
        a = np.array([int(win["asset_id"].iloc[-1])], dtype=np.int64)
        xt,mt,at = torch.from_numpy(x).to(device), torch.from_numpy(m).to(device), torch.from_numpy(a).to(device)
        model.eval();
        with torch.no_grad(): yh = model(xt,at,mt).squeeze(0).cpu().numpy()
        out = {"symbol": sym}
        for h,v in zip(HORIZONS, yh):
            lab=HLAB[h]; out[f"pred_{lab}_logret"]=float(v); out[f"pred_{lab}_ret"]=float(np.expm1(v))
        rows.append(out)
    df = pd.DataFrame(rows)
    logger.info(f"Point predictions generated: rows={len(df)}")
    return df


def score_quant(panel, FEATURES, scaler, model, HORIZONS, HLAB, QUANTS, device):
    logger.info("Scoring quantile predictions")
    rows = []
    q2i = {q: i for i, q in enumerate(QUANTS)}
    L = args.lookback
    for sym, g in panel.groupby(SYMBOL_COL):
        g = g.sort_values(TIME_COL)
        if len(g) < L: continue
        win = g.iloc[-L:]
        x = win[FEATURES].values.astype(np.float32)[None,:,:]; x = apply_scaler(x, scaler)
        m = win["is_trading_day"].values.astype(bool)[None,:]
        a = np.array([int(win["asset_id"].iloc[-1])], dtype=np.int64)
        xt,mt,at = torch.from_numpy(x).to(device), torch.from_numpy(m).to(device), torch.from_numpy(a).to(device)
        model.eval();
        with torch.no_grad(): yhq = model(xt,at,mt).squeeze(0).cpu().numpy()
        out = {"symbol": sym}
        for hi,h in enumerate(HORIZONS):
            lab=HLAB[h]
            for q in QUANTS:
                v=float(yhq[hi,q2i[q]]); out[f"pred_{lab}_p{int(q*100)}_logret"]=v; out[f"pred_{lab}_p{int(q*100)}_ret"]=float(np.expm1(v))
        rows.append(out)
    df = pd.DataFrame(rows)
    logger.info(f"Quantile predictions generated: rows={len(df)}")
    return df

def persist_topk_to_mongo(pred_df: pd.DataFrame, mongo_uri, dbname, coll_pred, top_k, outdir="outputs"):
    logger.info(
        f"Persisting top {top_k} predictions to MongoDB collection {dbname}.{coll_pred}"
    )
    os.makedirs(outdir, exist_ok=True)
    logger.info(pred_df.to_json())
    cli = mongo_client()
    as_of = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    out = pred_df.head(top_k).copy()
    out["as_of"] = as_of
    if len(out):
        cli[dbname][coll_pred].insert_many(out.to_dict(orient="records"))
        logger.info(f"Inserted {len(out)} docs into {dbname}.{coll_pred}")


def save_artifacts(model, scaler, asset2id, FEATURES, outdir="artifacts"):
    os.makedirs(outdir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(outdir, "model.pt"))
    joblib.dump({"scaler": scaler, "asset2id": asset2id, "features": FEATURES}, os.path.join(outdir, "prep.joblib"))
    logger.info(f"Saved artifacts → {outdir}")

# ==========================
# CLI & main
# ==========================

def parse_args():
    p = argparse.ArgumentParser(description="Custom forecaster (null‑safe)")
    # Mongo
    p.add_argument("--mongo-uri", default=os.getenv("MONGO_URI", "mongodb://admin:2059$tephan5203@94.130.171.244:27017/?authSource=admin&readPreference=secondaryPreferred&directConnection=true"))
    p.add_argument("--db", default=os.getenv("MONGO_DB", "PriceForecast"))
    p.add_argument("--coll-stock", default=os.getenv("COLL_STOCK", "DailyStockData"))
    p.add_argument("--coll-crypto", default=os.getenv("COLL_CRYPTO", "DailyCryptoData"))
    p.add_argument("--coll-fx", default=os.getenv("COLL_FX", "DailyCurrencyExchangeRates"))
    p.add_argument("--coll-news", default=os.getenv("COLL_NEWS", "NewsHeadlines"))
    p.add_argument("--coll-weather", default=os.getenv("COLL_WEATHER", "DailyWeatherData"))
    p.add_argument("--coll-pred", default=os.getenv("COLL_PRED", "PricePredictions"))
    # Horizons
    p.add_argument("--use-trading-days", action="store_true", default=os.getenv("USE_TRADING_DAYS", "false").lower()=="true")
    p.add_argument("--horizons-cal", type=str, default=os.getenv("HORIZONS_CAL", "7,30,182,365"))
    p.add_argument("--horizons-td", type=str, default=os.getenv("HORIZONS_TD", "5,21,126,252"))
    # Weather
    p.add_argument("--weather-pca", type=int, default=int(os.getenv("WEATHER_PCA", 12)))
    p.add_argument("--weather-agg", type=str, choices=["mean","median"], default=os.getenv("WEATHER_AGG", "mean"))
    p.add_argument("--fold-aware-weather-pca", action="store_true", default=os.getenv("FOLD_AWARE_WEATHER_PCA", "false").lower()=="true")
    # Model/training
    p.add_argument("--lookback", type=int, default=int(os.getenv("LOOKBACK", 180)))
    p.add_argument("--batch", type=int, default=int(os.getenv("BATCH", 256)))
    p.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", 15)))
    p.add_argument("--lr", type=float, default=float(os.getenv("LR", 7e-4)))
    p.add_argument("--dropout", type=float, default=float(os.getenv("DROPOUT", 0.15)))
    p.add_argument("--layers", type=int, default=int(os.getenv("LAYERS", 2)))
    p.add_argument("--hidden", type=int, default=int(os.getenv("HIDDEN", 128)))
    p.add_argument("--weight-decay", type=float, default=float(os.getenv("WEIGHT_DECAY", 1e-4)))
    p.add_argument("--seed", type=int, default=int(os.getenv("SEED", 42)))
    p.add_argument("--num-workers", type=int, default=int(os.getenv("NUM_WORKERS", 2)))
    # Split strategy
    p.add_argument("--walk-forward", action="store_true", default=os.getenv("WALK_FORWARD", "false").lower()=="true")
    p.add_argument("--train-span-days", type=int, default=int(os.getenv("TRAIN_SPAN_DAYS", 365*3)))
    p.add_argument("--val-span-days", type=int, default=int(os.getenv("VAL_SPAN_DAYS", 90)))
    p.add_argument("--step-days", type=int, default=int(os.getenv("STEP_DAYS", 90)))
    p.add_argument("--val-ratio", type=float, default=float(os.getenv("VAL_RATIO", 0.2)))
    # Quantiles
    p.add_argument("--use-quantiles", action="store_true", default=os.getenv("USE_QUANTILES", "false").lower()=="true")
    p.add_argument("--quantiles", type=str, default=os.getenv("QUANTILES", "0.5,0.9"))
    # News PCA cap
    p.add_argument("--news-pca-cap", type=int, default=int(os.getenv("NEWS_PCA_CAP", 32)))
    # Output / ranking
    p.add_argument("--top-k", type=int, default=int(os.getenv("TOP_K", 20)))
    p.add_argument("--rank-horizon", type=str, default=os.getenv("RANK_HORIZON", "1y"))
    p.add_argument("--rank-quantile", type=float, default=float(os.getenv("RANK_QUANTILE", 0.5)))
    p.add_argument("--ignore-days", type=int, default=int(os.getenv("IGNORE_DAYS", 0)))
    # Artifacts
    p.add_argument("--artifacts-dir", type=str, default=os.getenv("ARTIFACTS_DIR", "outputs"))
    return p.parse_args()


def quick_nan_report(X, name="X"):
    bad = ~np.isfinite(X)
    if bad.any():
        i = np.where(bad)
        logger.info(f"[WARN] Non-finite in {name}: count={bad.sum()} at first example/t/f={i[0][0]},{i[1][0]},{i[2][0]}")

def reduce_daily_duplicates(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g = g.sort_values(TIME_COL).dropna(subset=[TIME_COL])

    rules = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "quote_asset_volume": "sum",
        "num_trades": "sum",
        "taker_buy_base_vol": "sum",
        "taker_buy_quote_vol": "sum",
        "closePctChange": "last",
        "openPctChange": "last",
    }
    num_cols = g.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        rules.setdefault(c, "last")

    if g[TIME_COL].duplicated().any():
        cols = [TIME_COL] + [c for c in g.columns if c in rules]
        agg = g[cols].groupby(TIME_COL, as_index=False).agg({c: rules[c] for c in cols if c != TIME_COL})
        return agg
    else:
        return g

# --- the per-symbol worker (MUST be top-level) ---
def _process_one_symbol(sym: str,
                        g: pd.DataFrame,
                        cal: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Returns a processed per-symbol DataFrame; returns empty frame on error.
    """
    logger.info(f"Processing {sym}")
    try:
        # keep source before reindex drops it
        src = g["source"].iloc[0] if "source" in g.columns and len(g) else "unknown"

        # 0) collapse duplicate dates
        g = reduce_daily_duplicates(g)

        # 1) align to full daily calendar
        g = g.set_index(TIME_COL).reindex(cal)

        # 2) mark trading days
        g["is_trading_day"] = g["close"].notna().astype(int)

        # 3) fill & clamp prices (safe)
        for col in ["close", "open", "high", "low"]:
            g[col] = pd.to_numeric(g.get(col, np.nan), errors="coerce").ffill().bfill().fillna(0.0).clip(lower=EPS)

        # 4) activity features (0 on non-trading days)
        for col in ["volume","quote_asset_volume","num_trades","taker_buy_base_vol","taker_buy_quote_vol","closePctChange","openPctChange"]:
            g[col] = pd.to_numeric(g.get(col, 0.0), errors="coerce").fillna(0.0)
        for col in ["volume","quote_asset_volume","num_trades","taker_buy_base_vol","taker_buy_quote_vol"]:
            g[col] = g[col] * g["is_trading_day"].fillna(0)

        # 5) technicals (safe)
        logc = safe_log(g["close"].to_numpy())
        g["logret_1d"] = np.diff(logc, prepend=logc[0])
        g["vol_20d"]   = pd.Series(g["logret_1d"], index=g.index).rolling(20, min_periods=1).std().fillna(0.0).to_numpy()
        g["mom_63d"]   = logc - np.roll(logc, 63)
        g.iloc[:63, g.columns.get_loc("mom_63d")] = 0.0
        g["hl_spread"] = ((g["high"] - g["low"]) / np.maximum(g["close"], EPS)).astype(float)

        # 6) restore ids
        g[SYMBOL_COL] = sym
        g[TIME_COL]   = g.index
        g["source"]   = src

        return g.reset_index(drop=True)

    except Exception as e:
        # If you have a cross-process safe logger, use it; else print
        logger.exception(f"[worker] symbol={sym} failed: {e}")
        # Re-raise so the main process can also log the failure and skip the symbol
        raise

# --- orchestrate in parallel ---------------------
def _process_one_symbol_star(args: Tuple[str, pd.DataFrame, pd.DatetimeIndex]) -> pd.DataFrame:
    """Helper to unpack args for ``ProcessPoolExecutor.map``."""
    return _process_one_symbol(*args)


def process_all_symbols_parallel(
    prices: pd.DataFrame,
    cal: pd.DatetimeIndex,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """Parallelize per-symbol processing with a process pool.

    Any symbol that raises an exception during processing is logged and skipped
    so that a single bad symbol does not abort the entire workflow.
    """

    max_workers = max_workers or max(1, (os.cpu_count() or 2) - 1)
    logger.info(f"Processing {prices[SYMBOL_COL].nunique()} symbols in parallel")

    rows: List[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut_to_sym: Dict[object, str] = {}
        # submit at most ``max_workers`` tasks at a time to limit memory use
        for sym, grp in prices.groupby(SYMBOL_COL, sort=False):
            try:
                fut = ex.submit(_process_one_symbol_star, (sym, grp.copy(), cal))
            except Exception as e:
                logger.exception(f"[main] failed to submit symbol={sym}: {e}")
                continue
            fut_to_sym[fut] = sym
            if len(fut_to_sym) >= max_workers:
                done, _ = wait(fut_to_sym.keys(), return_when=FIRST_COMPLETED)
                for d in done:
                    sym_d = fut_to_sym.pop(d)
                    try:
                        rows.append(d.result())
                    except Exception as e:
                        logger.exception(f"[main] symbol={sym_d} failed: {e}")

        # drain any remaining futures
        for fut in as_completed(fut_to_sym):
            sym = fut_to_sym[fut]
            try:
                rows.append(fut.result())
            except Exception as e:
                logger.exception(f"[main] symbol={sym} failed: {e}")

    if not rows:
        logger.info("No rows returned from parallel processing")
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    logger.info(f"Parallel processing complete: rows={len(out)}")
    return out

# -------------------------------------------------
# Panel preparation helpers

def daily_calendar(prices: pd.DataFrame) -> pd.DatetimeIndex:
    logger.info("Generating daily calendar")
    first = prices[TIME_COL].min()
    last = prices[TIME_COL].max()
    return pd.date_range(first, last, freq="D")


def add_time_feats(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Adding time features")
    df["dow"] = df[TIME_COL].dt.dayofweek
    df["dom"] = df[TIME_COL].dt.day
    df["month"] = df[TIME_COL].dt.month
    logger.info("Added time features")
    return df


def prepare_panel(stock_df: pl.DataFrame,
                  crypto_df: pl.DataFrame,
                  fx_df: pl.DataFrame,
                  news_df: pl.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    logger.info("Preparing panel data")
    s = prep_stocks(stock_df)
    c = prep_crypto(crypto_df)
    s_pd = reduce_mem_usage(s.to_pandas() if hasattr(s, "to_pandas") else s, sparse_threshold=0.9)
    c_pd = reduce_mem_usage(c.to_pandas() if hasattr(c, "to_pandas") else c, sparse_threshold=0.9)
    if s_pd.empty and c_pd.empty:
        raise RuntimeError("No stock or crypto rows found.")
    prices = reduce_mem_usage(pd.concat([s_pd, c_pd], ignore_index=True), sparse_threshold=0.9)
    del s_pd, c_pd
    gc.collect()

    cal = daily_calendar(prices)
    panel = process_all_symbols_parallel(prices, cal, max_workers=8)
    del prices
    gc.collect()

    fx_norm = prep_fx(fx_df)

    fx_pd = reduce_mem_usage(fx_norm.to_pandas() if hasattr(fx_norm, "to_pandas") else fx_norm, sparse_threshold=0.9)
    panel = panel.merge(fx_pd, on=TIME_COL, how="left") if not fx_pd.empty else panel.assign(eur_fx_logret=0.0)
    del fx_pd
    gc.collect()

    newsg = prep_news_global(news_df)
    news_pd = reduce_mem_usage(newsg.to_pandas() if hasattr(newsg, "to_pandas") else newsg, sparse_threshold=0.9)
    if not news_pd.empty:
        panel = panel.merge(news_pd, on=TIME_COL, how="left")
    else:
        panel["news_sent_mean"] = 0.0
        panel["news_sent_max"] = 0.0
        panel["news_count"] = 0
        for i in range(8):
            panel[f"news_pca_{i}"] = 0.0
    del news_pd
    gc.collect()

    panel = add_time_feats(panel)
    panel = reduce_mem_usage(panel, sparse_threshold=0.9)

    num_cols = panel.select_dtypes(include=[np.number]).columns
    panel[num_cols] = (
        panel[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    )

    asset2id = {s: i for i, s in enumerate(sorted(panel[SYMBOL_COL].dropna().unique()))}
    panel["asset_id"] = panel[SYMBOL_COL].map(asset2id).astype(np.int32)
    logger.info(f"Prepared panel: shape={panel.shape}")
    return panel, asset2id

async def main(args):
    logger.info("[boot] starting main()")
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[env] Device: {device}")

    cal = [int(x) for x in args.horizons_cal.split(",") if x]
    td  = [int(x) for x in args.horizons_td.split(",") if x]
    if args.use_trading_days:
        HORIZONS = td;  HLAB = {5:"1w", 21:"1m",126:"6m",252:"1y"}
    else:
        HORIZONS = cal; HLAB = {7: "1w", 30:"1m",182:"6m",365:"1y"}


    logger.info("Connecting to MongoDB")
    cli = mongo_client(); db = cli[args.db]

    def iter_stocks_chunks(start_id=None):
        for df, last_id in iter_mongo_df_chunks(
                db.DailyStockData,
                query={"date" : {"$lte" : datetime.today().replace(hour = 0, minute = 0, second=0, microsecond=0) - timedelta(days=args.ignore_days)}},
                projection={"date": 1, "symbol": 1, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
                chunk_rows=100_000,
                start_id=start_id
        ):
            yield df, last_id

    def stock_schema(float_bits=32) -> pa.Schema:
        f = pa.float32() if float_bits == 32 else pa.float64()
        return pa.schema([
            pa.field("date", pa.timestamp("ns")),
            pa.field("symbol", pa.string()),
            pa.field("open", f),
            pa.field("high", f),
            pa.field("low", f),
            pa.field("close", f),
            pa.field("volume", f),
        ])

    def iter_crypto_chunks(start_id=None):
        for df, last_id in iter_mongo_df_chunks(
                db.DailyCryptoData,
                query={"date": {
                    "$lte": datetime.today().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=args.ignore_days)}},
                chunk_rows=100_000,
                start_id=start_id
        ):
            yield df, last_id

    def crypto_schema(float_bits=32) -> pa.Schema:
        f = pa.float32() if float_bits == 32 else pa.float64()
        return pa.schema([
            pa.field("date", pa.timestamp("ns")),
            pa.field("symbol", pa.string()),
            pa.field("open", f),
            pa.field("high", f),
            pa.field("low", f),
            pa.field("close", f),
            pa.field("volume", f),
            pa.field("quote_asset_volume", f),
            pa.field("num_trades", f),
            pa.field("taker_buy_base_vol", f),
            pa.field("taker_buy_quote_vol", f),
            pa.field("closePctChange", f),
            pa.field("openPctChange", f)
        ])

    def iter_weather_chunks(start_id=None):
        for df, last_id in iter_mongo_df_chunks(
                db.DailyWeatherData,
                query={"date": {
                    "$lte": datetime.today().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=args.ignore_days)}},
                chunk_rows=100_000,
                start_id=start_id
        ):
            yield df, last_id

    def weather_schema(float_bits=32) -> pa.Schema:
        f = pa.float32() if float_bits == 32 else pa.float64()
        return pa.schema([
            pa.field("date", pa.timestamp("ns")),
            pa.field("country", pa.string()),
            pa.field("station", pa.string()),
            pa.field("tavg", f),
            pa.field("tmin", f),
            pa.field("tmax", f),
            pa.field("prcp", f),
            pa.field("snow", f),
            pa.field("wdir", f),
            pa.field("wspd", f),
            pa.field("wpgt", f),
            pa.field("pres", f),
            pa.field("tsun", f)
        ])

    def news_schema() -> pa.Schema:
        return pa.schema([
            pa.field("date", pa.timestamp("ns")),
            pa.field("headline", pa.string()),
            # list<float32> (variable length)
            pa.field("sentiment", pa.list_(pa.float32())),
            # fixed_size_list<float32, 512> (exactly 512 elements per row)
            pa.field("embeddings", pa.list_(pa.float32(), 512)),
        ])

    def iter_news_chunks(start_id=None):
        for df, last_id in iter_mongo_df_chunks(
                db.NewsHeadlines,
                query={"date": {
                    "$lte": datetime.today().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=args.ignore_days)}},
                projection={"date": 1, "headline": 1, "sentiment": 1, "embeddings": 1},
                chunk_rows=20000,
                start_id=start_id
        ):
            # sanitize arrays
            df = df.with_columns(
                sanitize_list_column(df.get_column("sentiment"), np.float32, fixed_len=None)
                .alias("sentiment")
            )
            df = df.with_columns(
                sanitize_list_column(df.get_column("embeddings"), np.float32, fixed_len=512)
                .alias("embeddings")
            )
            yield df, last_id

    async def stage_news(out_path="data/NewsHeadlines.parquet", resume=True):
        schema = news_schema()
        # If the file exists, read its schema so we match order/types:
        target_schema = schema
        if os.path.exists(out_path):
            try:
                target_schema = pq.ParquetFile(out_path).schema_arrow
            except Exception:
                pass

        writer = None
        ckpt = out_path + ".ckpt"
        start_id = open(ckpt).read().strip() if (resume and os.path.exists(ckpt)) else None
        try:
            for i, (df, last_id) in enumerate(iter_news_chunks(start_id)):
                tbl = table_from_df_and_schema(df, target_schema)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, tbl.schema, compression="zstd")
                writer.write_table(tbl)
                with open(ckpt, "w") as f:
                    f.write(last_id)
                logger.info(f"[news] wrote chunk {i} rows={len(df)}")
        finally:
            if writer: writer.close()



    def iter_fx_chunks(start_id=None):
        for df, last_id in iter_mongo_df_chunks(
                db.DailyCurrencyExchangeRates,
                query={"date": {
                    "$lte": datetime.today().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=args.ignore_days)}},
                chunk_rows=200_000,
                start_id=start_id
        ):
            yield df, last_id

    def fx_schema(float_bits=32) -> pa.Schema:
        f = pa.float32() if float_bits == 32 else pa.float64()
        return pa.schema([
            pa.field("date", pa.timestamp("ns")),
            pa.field("base_currency", pa.string()),
            pa.field("AED", f), pa.field("AFN", f), pa.field("ALL", f), pa.field("AMD", f),
            pa.field("ANG", f), pa.field("AOA", f), pa.field("ARS", f), pa.field("AUD", f),
            pa.field("AWG", f), pa.field("AZN", f), pa.field("BAM", f), pa.field("BBD", f),
            pa.field("BDT", f), pa.field("BGN", f), pa.field("BHD", f), pa.field("BIF", f),
            pa.field("BMD", f), pa.field("BND", f), pa.field("BOB", f), pa.field("BRL", f),
            pa.field("BSD", f), pa.field("BTC", f), pa.field("BTN", f), pa.field("BWP", f),
            pa.field("BYN", f), pa.field("BYR", f), pa.field("BZD", f), pa.field("CAD", f),
            pa.field("CDF", f), pa.field("CHF", f), pa.field("CLF", f), pa.field("CLP", f),
            pa.field("CNY", f), pa.field("CNH", f), pa.field("COP", f), pa.field("CRC", f),
            pa.field("CUC", f), pa.field("CUP", f), pa.field("CVE", f), pa.field("CZK", f),
            pa.field("DJF", f), pa.field("DKK", f), pa.field("DOP", f), pa.field("DZD", f),
            pa.field("EGP", f), pa.field("ERN", f), pa.field("ETB", f), pa.field("EUR", f),
            pa.field("FJD", f), pa.field("FKP", f), pa.field("GBP", f), pa.field("GEL", f),
            pa.field("GGP", f), pa.field("GHS", f), pa.field("GIP", f), pa.field("GMD", f),
            pa.field("GNF", f), pa.field("GTQ", f), pa.field("GYD", f), pa.field("HKD", f),
            pa.field("HNL", f), pa.field("HRK", f), pa.field("HTG", f), pa.field("HUF", f),
            pa.field("IDR", f), pa.field("ILS", f), pa.field("IMP", f), pa.field("INR", f),
            pa.field("IQD", f), pa.field("IRR", f), pa.field("ISK", f), pa.field("JEP", f),
            pa.field("JMD", f), pa.field("JOD", f), pa.field("JPY", f), pa.field("KES", f),
            pa.field("KGS", f), pa.field("KHR", f), pa.field("KMF", f), pa.field("KPW", f),
            pa.field("KRW", f), pa.field("KWD", f), pa.field("KYD", f), pa.field("KZT", f),
            pa.field("LAK", f), pa.field("LBP", f), pa.field("LKR", f), pa.field("LRD", f),
            pa.field("LSL", f), pa.field("LTL", f), pa.field("LVL", f), pa.field("LYD", f),
            pa.field("MAD", f), pa.field("MDL", f), pa.field("MGA", f), pa.field("MKD", f),
            pa.field("MMK", f), pa.field("MNT", f), pa.field("MOP", f), pa.field("MRU", f),
            pa.field("MUR", f), pa.field("MVR", f), pa.field("MWK", f), pa.field("MXN", f),
            pa.field("MYR", f), pa.field("MZN", f), pa.field("NAD", f), pa.field("NGN", f),
            pa.field("NIO", f), pa.field("NOK", f), pa.field("NPR", f), pa.field("NZD", f),
            pa.field("OMR", f), pa.field("PAB", f), pa.field("PEN", f), pa.field("PGK", f),
            pa.field("PHP", f), pa.field("PKR", f), pa.field("PLN", f), pa.field("PYG", f),
            pa.field("QAR", f), pa.field("RON", f), pa.field("RSD", f), pa.field("RUB", f),
            pa.field("RWF", f), pa.field("SAR", f), pa.field("SBD", f), pa.field("SCR", f),
            pa.field("SDG", f), pa.field("SEK", f), pa.field("SGD", f), pa.field("SHP", f),
            pa.field("SLE", f), pa.field("SLL", f), pa.field("SOS", f), pa.field("SRD", f),
            pa.field("STD", f), pa.field("SVC", f), pa.field("SYP", f), pa.field("SZL", f),
            pa.field("THB", f), pa.field("TJS", f), pa.field("TMT", f), pa.field("TND", f),
            pa.field("TOP", f), pa.field("TRY", f), pa.field("TTD", f), pa.field("TWD", f),
            pa.field("TZS", f), pa.field("UAH", f), pa.field("UGX", f), pa.field("USD", f),
            pa.field("UYU", f), pa.field("UZS", f), pa.field("VES", f), pa.field("VND", f),
            pa.field("VUV", f), pa.field("WST", f), pa.field("XAF", f), pa.field("XAG", f),
            pa.field("XAU", f), pa.field("XCD", f), pa.field("XDR", f), pa.field("XOF", f),
            pa.field("XPF", f), pa.field("YER", f), pa.field("ZAR", f), pa.field("ZMK", f),
            pa.field("ZMW", f), pa.field("ZWL", f),
        ])

    await asyncio.gather(stage_collection_with_schema(
                                out_path="data/DailyStockData.parquet",
                                iter_chunks_fn=iter_stocks_chunks,
                                canonical_schema=stock_schema(32),  # used only if file doesn't exist yet
                                resume=True,
                                compression="zstd"),
        stage_collection_with_schema(out_path="data/DailyCryptoData.parquet",
                                     iter_chunks_fn=iter_crypto_chunks,
                                     canonical_schema=crypto_schema(32),  # used only if file doesn't exist yet
                                     resume=True,
                                     compression="zstd"),
        stage_collection_with_schema(out_path="data/DailyWeatherData.parquet",
                                     iter_chunks_fn=iter_weather_chunks,
                                     canonical_schema=weather_schema(32),  # used only if file doesn't exist yet
                                     resume=True,
                                     compression="zstd"),
        stage_news(),
        stage_collection_with_schema(out_path="data/DailyCurrencyExchangeRates.parquet",
                                     iter_chunks_fn=iter_fx_chunks,
                                     canonical_schema=fx_schema(32),  # used only if file doesn't exist yet
                                     resume=True,
                                     compression="zstd")
    )

    stock  = pl.read_parquet("data/DailyStockData.parquet")
    crypto = pl.read_parquet("data/DailyCryptoData.parquet")
    fx     = pl.read_parquet("data/DailyCurrencyExchangeRates.parquet")
    news   = pl.read_parquet("data/NewsHeadlines.parquet")
    wxraw  = pl.read_parquet("data/DailyWeatherData.parquet")

    logger.info(f"[data] rows: stock={len(stock)} crypto={len(crypto)} fx={len(fx)} news={len(news)} weather={len(wxraw)}")

    logger.info("Preparing panel")
    panel, asset2id = prepare_panel(stock, crypto, fx, news)
    del stock, crypto, fx, news
    gc.collect()

    if args.use_trading_days:
        panel, TARGET_COLS, MASK_COLS = add_trading_targets(panel, HORIZONS)
    else:
        panel, TARGET_COLS, MASK_COLS = add_calendar_targets(panel, HORIZONS)

    wx_daily = prep_weather_daily(wxraw, agg=args.weather_agg)
    del wxraw
    gc.collect()

    if not args.walk_forward or not args.fold_aware_weather_pca:
        panel, _, _, _ = merge_weather(panel, wx_daily, n_components=args.weather_pca, fit_dates=None)
        news_cols = [c for c in panel.columns if c.startswith("news_pca_")]
        wx_cols   = [c for c in panel.columns if c.startswith("wx_pca_")]
        FEATURES = [
            "logret_1d","vol_20d","mom_63d","hl_spread","volume","is_trading_day",
            "quote_asset_volume","num_trades","taker_buy_base_vol","taker_buy_quote_vol",
            "closePctChange","openPctChange",
            "eur_fx_logret","news_sent_mean","news_sent_max","news_count",
            "dow","dom","month",
        ] + news_cols + wx_cols
        X_raw, M, A, Y, D = build_sequences_multi(panel, args.lookback, FEATURES, TARGET_COLS, MASK_COLS, True)
        logger.info(f"[seq] X_raw={X_raw.shape} Y={Y.shape}")
        if not len(Y):
            raise RuntimeError("No training samples after sequence build; reduce lookback or check targets.")
        quick_nan_report(X_raw, "X_raw before scaling")
        # Replace any non-finite values produced upstream so downstream
        # scalers don't error out. This operates in-place to avoid copies.
        np.nan_to_num(X_raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if not args.walk_forward:
            logger.info("Not Walking Forward")
            tr_idx, va_idx = time_split_idx(len(Y), args.val_ratio)
            scaler = fit_scaler_on_train(X_raw[tr_idx])
            Xtr, Xva = apply_scaler(X_raw[tr_idx], scaler), apply_scaler(X_raw[va_idx], scaler)
            quick_nan_report(Xtr, "Xtr"); quick_nan_report(Xva, "Xva")
            tr_ds, va_ds = SeqDS(Xtr, M[tr_idx], A[tr_idx], Y[tr_idx]), SeqDS(Xva, M[va_idx], A[va_idx], Y[va_idx])
            tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,  num_workers=args.num_workers, pin_memory=True)
            va_dl = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            H = len(HORIZONS)
            if args.use_quantiles:
                QUANTS = [float(x) for x in args.quantiles.split(",") if x]
                model = PriceForecastQuantilesMulti(Xtr.shape[-1], len(asset2id), H, QUANTS, args.hidden, args.layers, args.dropout)
                model, _ = train_quant(model, tr_dl, va_dl, device, args.epochs, args.lr, args.weight_decay)
                pred_df = score_quant(panel, FEATURES, scaler, model, HORIZONS, HLAB, QUANTS, device)
                rank_col = f"pred_{args.rank_horizon}_p{int(args.rank_quantile*100)}_logret"
            else:
                model = PriceForecastMulti(Xtr.shape[-1], len(asset2id), H, args.hidden, args.layers, args.dropout)
                model, _ = train_point(model, tr_dl, va_dl, device, args.epochs, args.lr, args.weight_decay)
                pred_df = score_point(panel, FEATURES, scaler, model, HORIZONS, HLAB, device)
                rank_col = f"pred_{args.rank_horizon}_logret"
            pred_df = pred_df.sort_values(rank_col, ascending=False)
            logger.info(pred_df.head(20))
            persist_topk_to_mongo(pred_df, args.mongo_uri, args.db, args.coll_pred, args.top_k)
            save_artifacts(model, scaler, asset2id, FEATURES, outdir=args.artifacts_dir)
            del Xtr, Xva, tr_ds, va_ds, tr_dl, va_dl, X_raw, M, A, Y, D, panel, wx_daily, model, scaler
            gc.collect()
            return
        else:
            logger.info("Walking Forward")
            metrics = []
            pred_df = pd.DataFrame()
            for i, (tr_idx, va_idx) in enumerate(walk_forward_splits(D, args.train_span_days, args.val_span_days, args.step_days), 1):
                logger.info(f"\n[fold {i}] train={len(tr_idx)} val={len(va_idx)}")
                scaler = fit_scaler_on_train(X_raw[tr_idx])
                Xtr, Xva = apply_scaler(X_raw[tr_idx], scaler), apply_scaler(X_raw[va_idx], scaler)
                tr_ds, va_ds = SeqDS(Xtr, M[tr_idx], A[tr_idx], Y[tr_idx]), SeqDS(Xva, M[va_idx], A[va_idx], Y[va_idx])
                tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,  num_workers=args.num_workers, pin_memory=True)
                va_dl = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)
                H = len(HORIZONS)
                if args.use_quantiles:
                    QUANTS = [float(x) for x in args.quantiles.split(",") if x]
                    model = PriceForecastQuantilesMulti(Xtr.shape[-1], len(asset2id), H, QUANTS, args.hidden, args.layers, args.dropout)
                    model, best = train_quant(model, tr_dl, va_dl, device, args.epochs, args.lr, args.weight_decay)
                else:
                    model = PriceForecastMulti(Xtr.shape[-1], len(asset2id), H, args.hidden, args.layers, args.dropout)
                    model, best = train_point(model, tr_dl, va_dl, device, args.epochs, args.lr, args.weight_decay)
                metrics.append(best)
                logger.info(f"[walk-forward] val losses: {metrics}")

                if args.use_quantiles:
                    pred_df = pd.concat([pred_df, score_quant(panel, FEATURES, scaler, model, HORIZONS, HLAB, QUANTS, device)])
                    rank_col = f"pred_{args.rank_horizon}_p{int(args.rank_quantile*100)}_logret"
                else:
                    pred_df = pd.concat([pred_df, score_point(panel, FEATURES, scaler, model, HORIZONS, HLAB, device)])
                    rank_col = f"pred_{args.rank_horizon}_logret"
                pred_df = pred_df.sort_values(rank_col, ascending=False)
                logger.info(f"\n{pred_df.head(20)}")
                del Xtr, Xva, tr_ds, va_ds, tr_dl, va_dl
                gc.collect()
            persist_topk_to_mongo(pred_df, args.mongo_uri, args.db, args.coll_pred, args.top_k)
            save_artifacts(model, scaler, asset2id, FEATURES, outdir=args.artifacts_dir)
            del X_raw, M, A, Y, D, panel, wx_daily, model, scaler
            gc.collect()
            return
    else:
        # fold-aware weather PCA
        base_FEATURES = [
            "logret_1d","vol_20d","mom_63d","hl_spread","volume","is_trading_day",
            "quote_asset_volume","num_trades","taker_buy_base_vol","taker_buy_quote_vol",
            "closePctChange","openPctChange",
            "eur_fx_logret","news_sent_mean","news_sent_max","news_count",
            "dow","dom","month",
        ] + [c for c in panel.columns if c.startswith("news_pca_")]
        Xb, Mb, Ab, Yb, Db = build_sequences_multi(panel, args.lookback, base_FEATURES, TARGET_COLS, MASK_COLS, True)
        if not len(Yb): raise RuntimeError("No training samples in base sequences.")
        quick_nan_report(Xb, "Xb before scaling")
        np.nan_to_num(Xb, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        model = None
        scaler = None
        panel_wx_final = None
        H = len(HORIZONS)
        if args.use_quantiles: QUANTS = [float(x) for x in args.quantiles.split(",") if x]
        for i, (tr_idx, va_idx) in enumerate(walk_forward_splits(Db, args.train_span_days, args.val_span_days, args.step_days), 1):
            logger.info(f"\n[fold {i}] train={len(tr_idx)} val={len(va_idx)}")
            train_dates = pd.to_datetime(Db[tr_idx]).unique()
            panel_wx, _, _, _ = merge_weather(panel.copy(), wx_daily, n_components=args.weather_pca, fit_dates=train_dates)
            panel_wx_final = panel_wx
            wx_cols = [c for c in panel_wx.columns if c.startswith("wx_pca_")]
            FEATURES = base_FEATURES + wx_cols
            Xr, Mr, Ar, Yr, Dr = build_sequences_multi(panel_wx, args.lookback, FEATURES, TARGET_COLS, MASK_COLS, True)
            quick_nan_report(Xr, "Xr before scaling")
            np.nan_to_num(Xr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            tr2, va2 = next(walk_forward_splits(Dr, args.train_span_days, args.val_span_days, args.step_days))
            scaler = fit_scaler_on_train(Xr[tr2])
            Xtr, Xva = apply_scaler(Xr[tr2], scaler), apply_scaler(Xr[va2], scaler)
            tr_ds, va_ds = SeqDS(Xtr, Mr[tr2], Ar[tr2], Yr[tr2]), SeqDS(Xva, Mr[va2], Ar[va2], Yr[va2])
            tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,  num_workers=args.num_workers, pin_memory=True)
            va_dl = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            if args.use_quantiles:
                model = PriceForecastQuantilesMulti(Xtr.shape[-1], len(asset2id), H, QUANTS, args.hidden, args.layers, args.dropout)
                model, best = train_quant(model, tr_dl, va_dl, device, args.epochs, args.lr, args.weight_decay)
            else:
                model = PriceForecastMulti(Xtr.shape[-1], len(asset2id), H, args.hidden, args.layers, args.dropout)
                model, best = train_point(model, tr_dl, va_dl, device, args.epochs, args.lr, args.weight_decay)
            logger.info(f"[fold {i}] best val: {best:.6f}")
            del panel_wx, Xr, Mr, Ar, Yr, Dr, Xtr, Xva, tr_ds, va_ds, tr_dl, va_dl
            gc.collect()
        pred_df = score_quant(panel_wx_final, FEATURES, scaler, model, HORIZONS, HLAB, QUANTS, device) if args.use_quantiles else \
                   score_point(panel_wx_final, FEATURES, scaler, model, HORIZONS, HLAB, device)
        del panel_wx_final
        rank_col = (f"pred_{args.rank_horizon}_p{int(args.rank_quantile*100)}_logret" if args.use_quantiles else f"pred_{args.rank_horizon}_logret")
        pred_df = pred_df.sort_values(rank_col, ascending=False)
        logger.info(pred_df.head(20))
        persist_topk_to_mongo(pred_df, args.mongo_uri, args.db, args.coll_pred, args.top_k)
        save_artifacts(model, scaler, asset2id, FEATURES, outdir=args.artifacts_dir)
        del model, scaler, Xb, Mb, Ab, Yb, Db, panel, wx_daily
        gc.collect()
        return


if __name__ == "__main__":
    try:
        args = parse_args()
        asyncio.run(main(args))
    except Exception:
        logger.exception("An error occured during execution")
