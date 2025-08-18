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
import logging
import sys
print(f"[boot] Python: {sys.version}")

# Try import torch early so failures show before anything else
try:
    import torch
    print(f"[boot] torch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print("[boot][FATAL] torch import failed:", e)
    raise

import os
import copy
import argparse
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from pymongo import MongoClient
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import joblib

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)-45s] %(asctime)s [%(levelname)-8s]: %(message)s",
    stream=sys.stdout,
    force=True
)

logger = logging.getLogger(__name__)

# ==========================
# Global constants / utils
# ==========================
TIME_COL = "date"
SYMBOL_COL = "symbol"
EPS = 1e-6

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def safe_log(x: np.ndarray | pd.Series) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return np.log(np.clip(a, EPS, None))

# ==========================
# Mongo helpers
# ==========================

def mongo_client(uri: str) -> MongoClient:
    return MongoClient(uri)


def _load_coll(db, name: str, proj: Optional[dict] = None) -> pd.DataFrame:
    if not proj:
        docs = list(db[name].find({}).limit(1000))
    else:
        docs = list(db[name].find({}, proj).limit(1000))
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])
    return df

# ==========================
# Schema‑aware loaders
# ==========================

def to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)


def prep_stocks(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    for k in ["date","symbol","open","close","high","low","volume"]:
        if k not in d.columns: d[k] = np.nan
    d = d.rename(columns={"date": TIME_COL, "symbol": SYMBOL_COL})
    d[TIME_COL] = to_dt(d[TIME_COL])
    for c in ["open","close","high","low","volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=[TIME_COL, SYMBOL_COL])
    d = d.sort_values([SYMBOL_COL, TIME_COL])
    d["source"] = "stock"
    return d[[SYMBOL_COL, TIME_COL, "open","close","high","low","volume","source"]]


def prep_crypto(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    for k in [
        "date","symbol","open","high","low","close","volume",
        "quote_asset_volume","num_trades","taker_buy_base_vol","taker_buy_quote_vol",
        "closePctChange","openPctChange",
    ]:
        if k not in d.columns: d[k] = np.nan
    d = d.rename(columns={"date": TIME_COL, "symbol": SYMBOL_COL})
    d[TIME_COL] = to_dt(d[TIME_COL])
    for c in [
        "open","high","low","close","volume","quote_asset_volume","num_trades",
        "taker_buy_base_vol","taker_buy_quote_vol","closePctChange","openPctChange",
    ]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=[TIME_COL, SYMBOL_COL])
    d = d.sort_values([SYMBOL_COL, TIME_COL])
    d["source"] = "crypto"
    return d[[
        SYMBOL_COL, TIME_COL, "open","close","high","low","volume",
        "quote_asset_volume","num_trades","taker_buy_base_vol","taker_buy_quote_vol",
        "closePctChange","openPctChange","source"
    ]]


def prep_fx(df: pd.DataFrame) -> pd.DataFrame:
    """DailyCurrencyExchangeRates: date, base_currency, dynamic currency codes (USD, JPY...).
    Produces daily mean EUR FX log return across the dynamic columns.
    """
    if df.empty:
        return df
    d = df.copy()
    d[TIME_COL] = to_dt(d.get(TIME_COL, pd.Series(index=d.index)))
    d = d.dropna(subset=[TIME_COL]).sort_values(TIME_COL)
    rate_cols = [c for c in d.columns if c not in {TIME_COL, "base_currency"}]
    if not rate_cols:
        return pd.DataFrame({TIME_COL: d[TIME_COL], "eur_fx_logret": np.zeros(len(d))})
    for c in rate_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce").ffill().bfill().fillna(1.0).clip(lower=EPS)
    for c in rate_cols:
        v = d[c].values.astype(float)
        d[f"lr_{c}"] = np.r_[0.0, np.diff(safe_log(v))]
    lr_cols = [f"lr_{c}" for c in rate_cols]
    out = d[[TIME_COL] + lr_cols].copy()
    out["eur_fx_logret"] = out[lr_cols].mean(axis=1, skipna=True).fillna(0.0)
    return out[[TIME_COL, "eur_fx_logret"]]


def _extract_sentiment(val) -> float:
    # sentiment is array inside array; outer length 1
    if val is None:
        return 0.0
    try:
        arr = np.array(val, dtype=float).flatten()
        if arr.size == 0 or not np.isfinite(arr).any():
            return 0.0
        return float(np.nanmean(arr))
    except Exception:
        return 0.0


def prep_news_global(df: pd.DataFrame, pca_cap: int = 32) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d[TIME_COL] = to_dt(d.get(TIME_COL, pd.Series(index=d.index)))
    d = d.dropna(subset=[TIME_COL])
    d["sentiment_scalar"] = d.get("sentiment", 0).apply(_extract_sentiment) if "sentiment" in d.columns else 0.0
    def _to_vec(x):
        try:
            a = np.asarray(x, dtype=float).reshape(-1)
            if a.size == 0 or not np.isfinite(a).any():
                return np.empty((0,), dtype=float)
            return a
        except Exception:
            return np.empty((0,), dtype=float)
    d["emb_vec"] = d.get("embeddings", [[]]).apply(_to_vec) if "embeddings" in d.columns else [[]]

    rows = []
    for dt, g in d.groupby(TIME_COL):
        sent_mean = float(np.nanmean(g["sentiment_scalar"].values)) if len(g) else 0.0
        sent_max  = float(np.nanmax(g["sentiment_scalar"].values)) if len(g) else 0.0
        cnt = int(len(g))
        vecs = [v for v in g["emb_vec"].values if isinstance(v, np.ndarray) and v.size > 0]
        E = np.vstack(vecs) if vecs else np.empty((0, 0))
        rows.append({TIME_COL: dt, "news_sent_mean": sent_mean, "news_sent_max": sent_max, "news_count": cnt, "E": E})
    agg = pd.DataFrame(rows)

    if (not agg.empty) and agg["E"].apply(lambda a: isinstance(a, np.ndarray) and a.size > 0).any():
        stacks = [a for a in agg["E"].values if isinstance(a, np.ndarray) and a.size > 0]
        X = np.vstack(stacks)
        comp_dim = min(pca_cap, X.shape[1])
        ipca = IncrementalPCA(n_components=comp_dim)
        chunk = 10000
        for i in range(0, X.shape[0], chunk):
            ipca.partial_fit(X[i:i+chunk])
        def _reduce(a):
            if not isinstance(a, np.ndarray) or a.size == 0:
                return np.zeros(comp_dim, dtype=float)
            return ipca.transform(a).mean(axis=0)
        Z = agg["E"].apply(_reduce)
        Zdf = pd.DataFrame(Z.tolist(), columns=[f"news_pca_{i}" for i in range(comp_dim)])
        agg = pd.concat([agg.drop(columns=["E"]), Zdf], axis=1)
    else:
        for i in range(8): agg[f"news_pca_{i}"] = 0.0
        agg = agg.drop(columns=["E"]) if "E" in agg.columns else agg

    return agg

# ==========================
# Panel construction
# ==========================

def daily_calendar(prices: pd.DataFrame) -> pd.DatetimeIndex:
    first = prices[TIME_COL].min(); last = prices[TIME_COL].max()
    return pd.date_range(first, last, freq="D")


def add_time_feats(df: pd.DataFrame) -> pd.DataFrame:
    df["dow"] = df[TIME_COL].dt.dayofweek
    df["dom"] = df[TIME_COL].dt.day
    df["month"] = df[TIME_COL].dt.month
    return df


def prepare_panel(stock_df, crypto_df, fx_df, news_df) -> Tuple[pd.DataFrame, Dict[str,int]]:
    s = prep_stocks(stock_df); c = prep_crypto(crypto_df)
    if s.empty and c.empty:
        raise RuntimeError("No stock or crypto rows found.")
    prices = pd.concat([s, c], ignore_index=True)

    cal = daily_calendar(prices)
    rows = []
    for sym, g in prices.groupby(SYMBOL_COL):
        g = g.set_index(TIME_COL).reindex(cal)
        g["is_trading_day"] = g["close"].notna().astype(int)
        # Fill and clamp prices
        for col in ["close","open","high","low"]:
            g[col] = pd.to_numeric(g.get(col, np.nan), errors="coerce").ffill().bfill().fillna(0.0).clip(lower=EPS)
        # Activity features (0 on non-trading days)
        for col in ["volume","quote_asset_volume","num_trades","taker_buy_base_vol","taker_buy_quote_vol","closePctChange","openPctChange"]:
            g[col] = pd.to_numeric(g.get(col, 0.0), errors="coerce").fillna(0.0)
        for col in ["volume","quote_asset_volume","num_trades","taker_buy_base_vol","taker_buy_quote_vol"]:
            g[col] = g[col] * g["is_trading_day"].fillna(0)
        # Technicals (safe)
        logc = safe_log(g["close"].to_numpy())
        g["logret_1d"] = np.diff(logc, prepend=logc[0])
        g["vol_20d"] = pd.Series(g["logret_1d"], index=g.index).rolling(20, min_periods=1).std().fillna(0.0).to_numpy()
        g["mom_63d"] = logc - np.roll(logc, 63)
        g.iloc[:63, g.columns.get_loc("mom_63d")] = 0.0
        g["hl_spread"] = ((g["high"] - g["low"]) / np.maximum(g["close"], EPS)).astype(float)
        # Restore ids
        g[SYMBOL_COL] = sym; g[TIME_COL] = g.index
        src = prices.loc[prices[SYMBOL_COL]==sym, "source"].iloc[0]
        g["source"] = src
        rows.append(g.reset_index(drop=True))
    panel = pd.concat(rows, ignore_index=True)

    # FX merge
    fx_norm = prep_fx(fx_df)
    panel = panel.merge(fx_norm, on=TIME_COL, how="left") if not fx_norm.empty else panel.assign(eur_fx_logret=0.0)

    # Global news by date
    newsg = prep_news_global(news_df)
    if not newsg.empty:
        panel = panel.merge(newsg, on=TIME_COL, how="left")
    else:
        panel["news_sent_mean"] = 0.0; panel["news_sent_max"] = 0.0; panel["news_count"] = 0
        for i in range(8): panel[f"news_pca_{i}"] = 0.0

    panel = add_time_feats(panel)

    # Purge non-finite numerics (features only)
    num_cols = panel.select_dtypes(include=[np.number]).columns
    panel[num_cols] = panel[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    asset2id = {s: i for i, s in enumerate(sorted(panel[SYMBOL_COL].dropna().unique()))}
    panel["asset_id"] = panel[SYMBOL_COL].map(asset2id).astype(int)
    return panel, asset2id

# ==========================
# Weather → PCA → merge
# ==========================

def prep_weather_daily(weather_df: pd.DataFrame, agg: str = "mean") -> pd.DataFrame:
    if weather_df.empty: return pd.DataFrame(columns=[TIME_COL])
    w = weather_df.copy(); w[TIME_COL] = to_dt(w.get(TIME_COL, pd.Series(index=w.index)))
    w = w.dropna(subset=[TIME_COL])
    wx_num = [c for c in ["tavg","tmin","tmax","prcp","snow","wdir","wspd","wpgt","pres","tsun"] if c in w.columns]
    for c in wx_num: w[c] = pd.to_numeric(w[c], errors="coerce")
    if agg == "median": wagg = w.groupby(TIME_COL)[wx_num].median().reset_index()
    else: wagg = w.groupby(TIME_COL)[wx_num].mean().reset_index()
    wagg = wagg.sort_values(TIME_COL).rename(columns={c: f"wx_{c}" for c in wx_num})
    return wagg


def fit_weather_pca(weather_daily: pd.DataFrame, n_components: int, fit_dates: Optional[List[pd.Timestamp]] = None):
    wx_cols = [c for c in weather_daily.columns if c.startswith("wx_")]
    if not wx_cols: return None, None, [], []
    df = weather_daily.copy()
    if fit_dates is not None:
        df = df[df[TIME_COL].isin(pd.to_datetime(fit_dates))]
        if df.empty: return None, None, wx_cols, []
    sc = StandardScaler(); Xfit = sc.fit_transform(df[wx_cols].values)
    pca = IncrementalPCA(n_components=min(n_components, Xfit.shape[1]))
    for i in range(0, Xfit.shape[0], 20000): pca.partial_fit(Xfit[i:i+20000])
    names = [f"wx_pca_{i}" for i in range(pca.n_components_)]
    return pca, sc, wx_cols, names


def transform_weather_pca(weather_daily: pd.DataFrame, pca, sc, wx_cols: List[str], names: List[str]):
    if pca is None or not wx_cols:
        out = weather_daily[[TIME_COL]].copy()
        for n in names: out[n] = 0.0
        return out
    X = sc.transform(weather_daily[wx_cols].values); Z = pca.transform(X)
    out = weather_daily[[TIME_COL]].copy()
    for i, n in enumerate(names): out[n] = Z[:, i].astype(np.float32)
    return out


def merge_weather(panel: pd.DataFrame, weather_daily: pd.DataFrame, n_components: int, fit_dates: Optional[List[pd.Timestamp]] = None):
    if weather_daily.empty:
        names = [f"wx_pca_{i}" for i in range(n_components)]
        for n in names: panel[n] = 0.0
        return panel, None, None, names
    pca, sc, wx_cols, names = fit_weather_pca(weather_daily, n_components, fit_dates)
    wxz = transform_weather_pca(weather_daily, pca, sc, wx_cols, names)
    out = panel.merge(wxz, on=TIME_COL, how="left")
    for n in names: out[n] = out[n].fillna(0.0)
    return out, pca, sc, names

# ==========================
# Targets
# ==========================

def add_calendar_targets(panel: pd.DataFrame, horizons_days: List[int]):
    panel = panel.sort_values([SYMBOL_COL, TIME_COL]).copy()
    tcols, mcols = [], []
    for h in horizons_days:
        tcol, mcol = f"target_{h}d", f"mask_{h}d"
        panel[tcol] = panel.groupby(SYMBOL_COL)["close"].transform(lambda s: (safe_log(s.shift(-h)) - safe_log(s)))
        panel[mcol] = panel[tcol].replace([np.inf,-np.inf], np.nan).notna().astype(int)
        tcols.append(tcol); mcols.append(mcol)
    return panel, tcols, mcols


def add_trading_targets(panel: pd.DataFrame, horizons_td: List[int]):
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
    return panel2, tcols, mcols

# ==========================
# Sequences / scaling / splits
# ==========================

def build_sequences_multi(panel: pd.DataFrame, lookback: int, features: List[str], target_cols: List[str], mask_cols: List[str], require_all: bool = True):
    panel = panel.sort_values([SYMBOL_COL, TIME_COL])
    X, M, A, Y, D = [], [], [], [], []
    H = len(target_cols)
    for sym, g in panel.groupby(SYMBOL_COL):
        g = g.reset_index(drop=True)
        for t in range(lookback, len(g)):
            y = g.loc[t, target_cols].values.astype(np.float32)
            masks = g.loc[t, mask_cols].values.astype(int)
            if require_all:
                if not np.isfinite(y).all() or masks.sum() < H: continue
            else:
                if not np.isfinite(y).any(): continue
            win = g.iloc[t-lookback:t]
            if len(win) < lookback: continue
            X.append(win[features].values.astype(np.float32))
            M.append(win["is_trading_day"].values.astype(bool))
            A.append(int(win["asset_id"].iloc[-1]))
            Y.append(y)
            D.append(pd.Timestamp(g.loc[t, TIME_COL]).to_datetime64())
    X = np.stack(X) if len(X) else np.zeros((0, lookback, len(features)), np.float32)
    M = np.stack(M) if len(M) else np.zeros((0, lookback), bool)
    A = np.array(A, np.int64)
    Y = np.stack(Y) if len(Y) else np.zeros((0, H), np.float32)
    D = np.array(D)
    return X, M, A, Y, D


def time_split_idx(n: int, val_ratio: float = 0.2):
    val_n = int(n * val_ratio); idx = np.arange(n)
    return idx[: n - val_n], idx[n - val_n :]


def walk_forward_splits(dates: np.ndarray, train_span_days=365*3, val_span_days=90, step_days=90):
    dates = pd.to_datetime(dates); start, end = dates.min(), dates.max()
    anchor = start + pd.Timedelta(days=train_span_days)
    while True:
        train_end = anchor; val_end = train_end + pd.Timedelta(days=val_span_days)
        if train_end > end or val_end > end: break
        tr = ((dates > train_end - pd.Timedelta(days=train_span_days)) & (dates <= train_end)).to_numpy()
        va = ((dates > train_end) & (dates <= val_end)).to_numpy()
        tr_idx, va_idx = np.nonzero(tr)[0], np.nonzero(va)[0]
        if len(tr_idx) and len(va_idx): yield tr_idx, va_idx
        anchor += pd.Timedelta(days=step_days)


def fit_scaler_on_train(Xtr: np.ndarray) -> StandardScaler:
    if Xtr.size == 0: raise RuntimeError("Empty training set after filtering.")
    N, T, F = Xtr.shape; sc = StandardScaler().fit(Xtr.reshape(N*T, F)); return sc


def apply_scaler(X: np.ndarray, sc: StandardScaler) -> np.ndarray:
    if X.size == 0: return X
    N, T, F = X.shape; return sc.transform(X.reshape(N*T, F)).reshape(N, T, F)

# ==========================
# Models & training
# ==========================
class AttentionPool(nn.Module):
    def __init__(self, d):
        super().__init__(); self.attn = nn.Linear(d, 1)
    def forward(self, x, mask=None):
        s = self.attn(x).squeeze(-1)
        if mask is not None: s = s.masked_fill(~mask.bool(), float('-inf'))
        w = torch.softmax(s, dim=-1).unsqueeze(-1); return (x * w).sum(1), w.squeeze(-1)

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
        logger.info("[FATAL] Non-finite in X batch"); idx = torch.where(~torch.isfinite(x)); logger.info("indices:", [t[:5].tolist() for t in idx]); raise SystemExit(1)
    if torch.isnan(y).any() or torch.isinf(y).any():
        logger.info("[FATAL] Non-finite in Y batch"); idx = torch.where(~torch.isfinite(y)); logger.info("indices:", [t[:5].tolist() for t in idx]); raise SystemExit(1)


def train_point(model, tr_dl, va_dl, device="cpu", epochs=10, lr=7e-4, weight_decay=1e-4, patience=5):
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
    model.load_state_dict(best_state); return model, best


def train_quant(model, tr_dl, va_dl, device="cpu", epochs=10, lr=7e-4, weight_decay=1e-4, patience=5):
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
    model.load_state_dict(best_state); return model, best

# ==========================
# Scoring & persistence
# ==========================

def score_point(panel, FEATURES, scaler, model, HORIZONS, HLAB, device):
    rows = []; L = args.lookback
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
    return pd.DataFrame(rows)


def score_quant(panel, FEATURES, scaler, model, HORIZONS, HLAB, QUANTS, device):
    rows = []; q2i = {q:i for i,q in enumerate(QUANTS)}; L = args.lookback
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
    return pd.DataFrame(rows)


def persist_topk_to_mongo(pred_df, mongo_uri, dbname, coll_pred, top_k):
    cli = mongo_client(mongo_uri); as_of = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    out = pred_df.head(top_k).copy(); out["as_of"]=as_of
    if len(out): cli[dbname][coll_pred].insert_many(out.to_dict(orient="records")); logger.info(f"Inserted {len(out)} docs into {dbname}.{coll_pred}")


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
    p.add_argument("--mongo-uri", default=os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    p.add_argument("--db", default=os.getenv("MONGO_DB", "PriceForecast"))
    p.add_argument("--coll-stock", default=os.getenv("COLL_STOCK", "DailyStockData"))
    p.add_argument("--coll-crypto", default=os.getenv("COLL_CRYPTO", "DailyCryptoData"))
    p.add_argument("--coll-fx", default=os.getenv("COLL_FX", "DailyCurrencyExchangeRates"))
    p.add_argument("--coll-news", default=os.getenv("COLL_NEWS", "NewsHeadlines"))
    p.add_argument("--coll-weather", default=os.getenv("COLL_WEATHER", "DailyWeatherData"))
    p.add_argument("--coll-pred", default=os.getenv("COLL_PRED", "Predictions"))
    # Horizons
    p.add_argument("--use-trading-days", action="store_true", default=os.getenv("USE_TRADING_DAYS", "true").lower()=="true")
    p.add_argument("--horizons-cal", type=str, default=os.getenv("HORIZONS_CAL", "30,182,365"))
    p.add_argument("--horizons-td", type=str, default=os.getenv("HORIZONS_TD", "21,126,252"))
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
    # Artifacts
    p.add_argument("--artifacts-dir", type=str, default=os.getenv("ARTIFACTS_DIR", "artifacts"))
    return p.parse_args()


def quick_nan_report(X, name="X"):
    bad = ~np.isfinite(X)
    if bad.any():
        i = np.where(bad)
        logger.info(f"[WARN] Non-finite in {name}: count={bad.sum()} at first example/t/f=", (i[0][0], i[1][0], i[2][0]))


def main(args):
    logger.info("[boot] starting main()")
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[env] Device: {device}")

    cal = [int(x) for x in args.horizons_cal.split(",") if x]
    td  = [int(x) for x in args.horizons_td.split(",") if x]
    if args.use_trading_days:
        HORIZONS = td;  HLAB = {21:"1m",126:"6m",252:"1y"}
    else:
        HORIZONS = cal; HLAB = {30:"1m",182:"6m",365:"1y"}

    cli = mongo_client(args.mongo_uri); db = cli[args.db]
    stock  = _load_coll(db, args.coll_stock)
    crypto = _load_coll(db, args.coll_crypto)
    fx     = _load_coll(db, args.coll_fx)
    news   = _load_coll(db, args.coll_news)
    wxraw  = _load_coll(db, args.coll_weather)

    logger.info(f"[data] rows: stock={len(stock)} crypto={len(crypto)} fx={len(fx)} news={len(news)} weather={len(wxraw)}")

    panel, asset2id = prepare_panel(stock, crypto, fx, news)

    if args.use_trading_days:
        panel, TARGET_COLS, MASK_COLS = add_trading_targets(panel, HORIZONS)
    else:
        panel, TARGET_COLS, MASK_COLS = add_calendar_targets(panel, HORIZONS)

    wx_daily = prep_weather_daily(wxraw, agg=args.weather_agg)

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
        if not args.walk_forward:
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
            return
        else:
            metrics = []
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
            logger.info("[walk-forward] val losses:", metrics)
            if args.use_quantiles:
                pred_df = score_quant(panel, FEATURES, scaler, model, HORIZONS, HLAB, QUANTS, device)
                rank_col = f"pred_{args.rank_horizon}_p{int(args.rank_quantile*100)}_logret"
            else:
                pred_df = score_point(panel, FEATURES, scaler, model, HORIZONS, HLAB, device)
                rank_col = f"pred_{args.rank_horizon}_logret"
            pred_df = pred_df.sort_values(rank_col, ascending=False)
            logger.info(pred_df.head(20))
            persist_topk_to_mongo(pred_df, args.mongo_uri, args.db, args.coll_pred, args.top_k)
            save_artifacts(model, scaler, asset2id, FEATURES, outdir=args.artifacts_dir)
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
        models, scalers = [], []
        H = len(HORIZONS)
        if args.use_quantiles: QUANTS = [float(x) for x in args.quantiles.split(",") if x]
        for i, (tr_idx, va_idx) in enumerate(walk_forward_splits(Db, args.train_span_days, args.val_span_days, args.step_days), 1):
            logger.info(f"\n[fold {i}] train={len(tr_idx)} val={len(va_idx)}")
            train_dates = pd.to_datetime(Db[tr_idx]).unique()
            panel_wx, _, _, _ = merge_weather(panel.copy(), wx_daily, n_components=args.weather_pca, fit_dates=train_dates)
            wx_cols = [c for c in panel_wx.columns if c.startswith("wx_pca_")]
            FEATURES = base_FEATURES + wx_cols
            Xr, Mr, Ar, Yr, Dr = build_sequences_multi(panel_wx, args.lookback, FEATURES, TARGET_COLS, MASK_COLS, True)
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
            models.append(model); scalers.append(scaler)
        model, scaler = models[-1], scalers[-1]
        pred_df = score_quant(panel_wx, FEATURES, scaler, model, HORIZONS, HLAB, QUANTS, device) if args.use_quantiles else \
                   score_point(panel_wx, FEATURES, scaler, model, HORIZONS, HLAB, device)
        rank_col = (f"pred_{args.rank_horizon}_p{int(args.rank_quantile*100)}_logret" if args.use_quantiles else f"pred_{args.rank_horizon}_logret")
        pred_df = pred_df.sort_values(rank_col, ascending=False)
        logger.info(pred_df.head(20))
        persist_topk_to_mongo(pred_df, args.mongo_uri, args.db, args.coll_pred, args.top_k)
        save_artifacts(model, scaler, asset2id, FEATURES, outdir=args.artifacts_dir)
        return


if __name__ == "__main__":
    args = parse_args()
    main(args)
