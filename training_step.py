import copy
import datetime
import gc
import logging
import os
import sys
from argparse import Namespace
from typing import List

import pandas as pd
import numpy as np
import heapq, math
import duckdb
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.utils.data import IterableDataset

from util.constants import SYMBOL_COL, TIME_COL, EPS
from util.data_utils import mongo_client
from util.diagnostics import setup_diagnostics, disable_diagnostics, set_seed, __log
import torch.nn as nn
from util.multimodal_model import PriceEncoder, FXEncoder, MultiModalPriceForecast

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)-25s] %(asctime)s [%(levelname)-8s]: %(message)s",
    stream=sys.stdout,
    force=True
)

__log = logging.getLogger(__name__)

args: Namespace

_DUCK_CON = None

def _get_duck():
    global _DUCK_CON
    if _DUCK_CON is None:                     # each worker gets its own
        _DUCK_CON = duckdb.connect()
        _DUCK_CON.execute("PRAGMA threads=2;")
    return _DUCK_CON

def worker_init_fn(_):
    # force creation inside the worker process
    global _DUCK_CON
    _DUCK_CON = duckdb.connect()
    _DUCK_CON.execute("PRAGMA threads=2;")

class SeqDS(Dataset):
    def __init__(self, X, M, A, Y): self.X=torch.from_numpy(X); self.M=torch.from_numpy(M); self.A=torch.from_numpy(A); self.Y=torch.from_numpy(Y)
    def __len__(self): return len(self.Y)
    def __getitem__(self,i): return self.X[i], self.M[i], self.A[i], self.Y[i]

def build_sequences_multi_horizon(panel: pd.DataFrame,
                                 lookback: int,
                                 features: List[str],
                                 target_cols: List[str],
                                 mask_cols: List[str],
                                 require_all: bool = True):
    __log.info("Building Sequences for multi Horizon")

    H = len(target_cols)
    total = 0
    for sym, g in panel.groupby(SYMBOL_COL, observed=False):
        __log.info(f"Processing symbol {sym}")
        g = g.reset_index(drop=True)
        target_arr = g[target_cols].to_numpy(np.float32, copy=True)
        masks_arr = g[mask_cols].to_numpy(int, copy=True)


        for t in range(lookback, len(g)):
            y = target_arr[t]
            masks = masks_arr[t]

            if require_all:
                if not np.isfinite(y).all or masks.sum() < H:
                    continue
            else:
                if not np.isfinite(y).any():
                    continue
            total += 1

        del target_arr, masks_arr
        gc.collect()

    __log.info(f"Total sequences to build: {total}")

    X_Out = np.empty((total, lookback, len(features)), dtype=np.float32)
    M_Out = np.empty((total, lookback), dtype = bool)
    A_Out = np.empty((total, ), dtype=np.int64)
    Y_Out = np.empty((total, H), dtype=np.float32)
    D_Out = np.empty((total, ), dtype="datetime64[ns]")

    idx = 0

    for sym, g in panel.groupby(SYMBOL_COL, observed=False):
        __log.info(f"Building sequence for {sym}")

        g = g.reset_index(drop = True)

        feat_arr = g[features].to_numpy(np.float32, copy=True)
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

            win_slice = slice(t - (lookback), t)
            win_mask = win_slice
            if not win_mask:
                continue

            X_Out[idx] = feat_arr[win_slice]
            M_Out[idx] = win_mask
            A_Out[idx] = int(asset_ids[t - 1])
            Y_Out[idx] = y
            D_Out[idx] = dates_arr[t]
            idx += 1

        # explicit cleanup of large arrays for each symbol
        del feat_arr, asset_ids, targets_arr, masks_arr, dates_arr
        gc.collect()

    __log.info(f"Build sequences: X={X_Out.shape} Y={Y_Out.shape}")
    return X_Out, M_Out, A_Out, Y_Out, D_Out


def build_target_and_mask(horizon_days: List[int]):
    tcols, mcols = [], []
    for h in horizon_days:
        tcols.append(f"target_{h}d")
        mcols.append(f"mask_{h}d")

    return tcols, mcols

def walk_forward_splits(dates,
                        train_span_days=365*5,
                        val_span_days = 180,
                        step_days= 180):
    __log.info(
        f"Generating walk-forward splits train_span={train_span_days} val_span={val_span_days} step={step_days}"
    )

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
            __log.info(
                f"[split {i}] train={tr_idx.size} val={va_idx.size} anchor={anchor}"
            )
            yield tr_idx, va_idx
            i += 1

        anchor = anchor + step

def fit_scaler_on_trai(Xtr: np.ndarray):
    if Xtr.size == 0:
        raise RuntimeError("Empty training set after filtering")
    __log.info(f"Fitting scaler on data; shape={Xtr.shape}")
    N, T, F = Xtr.shape
    sc = StandardScaler().fit(Xtr.reshape(N * T, F))
    __log.info("Scaler fitted")

    return sc

def apply_scaler(X: np.ndarray, sc: StandardScaler) -> np.ndarray:
    if X.size == 0:
        __log.info("Apply scaler received empty array")
        return X

    __log.info(f"Applying scaler to array; shape={X.shape}")
    N, T, F = X.shape

    return sc.transform(X.reshape(N *T, F)).reshape(N, T, F)

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


class ZeroWeatherEncoder(nn.Module):
    """Placeholder encoder returning zeros for weather modality."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return torch.zeros(x.size(0), self.d_model, device=x.device)


class ZeroNewsEncoder(nn.Module):
    """Placeholder encoder returning zeros for news modality."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, tokens: dict, mask: torch.Tensor | None = None) -> torch.Tensor:
        device = tokens["input_ids"].device
        B = tokens["input_ids"].size(0)
        return torch.zeros(B, self.d_model, device=device)

def _first_batch_nan_guard(x, y):
    if torch.isnan(x).any() or torch.isinf(x).any():
        __log.info("[FATAL] Non-finite in X batch"); idx = torch.where(~torch.isfinite(x)); __log.info(f"indices: {[t[:5].tolist() for t in idx]}"); raise SystemExit(1)
    if torch.isnan(y).any() or torch.isinf(y).any():
        __log.info("[FATAL] Non-finite in Y batch"); idx = torch.where(~torch.isfinite(y)); __log.info(f"indices: {[t[:5].tolist() for t in idx]}"); raise SystemExit(1)

_last_tl = None
_last_vl = None

def train_point(model, tr_dl, va_dl, device="cpu", epochs=10, lr=7e-4, weight_decay=1e-4, patience=2,
                price_idx=None, fx_idx=None):
    __log.info("Starting point training")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss()
    best = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    bad = 0
    model.to(device)

    for ep in range(1, epochs + 1):
        model.train()
        tl_sum = 0.0
        tl_den = 0
        first = True

        for x, m, a, y in tqdm(tr_dl, desc=f"Training Epoch {ep}/{epochs}"):
            x, m, a, y = x.to(device), m.to(device), a.to(device), y.to(device)
            price_x = x[:, :, price_idx]
            fx_x = x[:, :, fx_idx]
            weather_x = torch.zeros(price_x.size(0), price_x.size(1), 1, device=device)
            news_tokens = {
                "input_ids": torch.zeros(price_x.size(0), price_x.size(1), 1, dtype=torch.long, device=device),
                "attention_mask": torch.zeros(price_x.size(0), price_x.size(1), 1, dtype=torch.long, device=device),
            }
            if first:
                _first_batch_nan_guard(x, y)
                first = False

            opt.zero_grad()
            pred = model(price_x, a, m, fx_x, m, weather_x, m, news_tokens, m)
            loss = loss_fn(pred, y)
            if torch.isnan(loss) or torch.isinf(loss):
                __log.info("[FATAL] loss non-finite"); raise SystemExit(1)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

            bsz = y.size(0)
            tl_sum += loss.item() * bsz
            tl_den += bsz

        tl = tl_sum / tl_den if tl_den else float("nan")

        # ---- validation ----
        model.eval()
        vl_sum = 0.0
        val_seen = 0

        with torch.no_grad():
            for x, m, a, y in tqdm(va_dl, f"Validating Batch"):
                x, m, a, y = x.to(device), m.to(device), a.to(device), y.to(device)
                price_x = x[:, :, price_idx]
                fx_x = x[:, :, fx_idx]
                weather_x = torch.zeros(price_x.size(0), price_x.size(1), 1, device=device)
                news_tokens = {
                    "input_ids": torch.zeros(price_x.size(0), price_x.size(1), 1, dtype=torch.long, device=device),
                    "attention_mask": torch.zeros(price_x.size(0), price_x.size(1), 1, dtype=torch.long, device=device),
                }
                vloss = nn.functional.smooth_l1_loss(
                    model(price_x, a, m, fx_x, m, weather_x, m, news_tokens, m), y
                )
                vl_sum += vloss.item() * y.size(0)
                val_seen += y.size(0)

        vl = vl_sum / val_seen if val_seen > 0 else float("inf")
        if val_seen == 0:
            __log.warning("Validation set is empty, setting val=inf")

        __log.info(f"Train {tl:.5f} | Val {vl:.5f}")

        if vl < best - 1e-4:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad > patience:
                __log.info("Early stop")
                break

    model.load_state_dict(best_state)
    __log.info("Finished point training")
    return model, best


def persist_topk_to_mongo(pred_df: pd.DataFrame, top_k = 25):
    __log.info(pred_df.head(top_k))

    cli = mongo_client()
    as_of = datetime.datetime.now().replace(hour=0,minute=0,second=0,microsecond=0)
    out = pred_df.head(top_k).copy()
    out["as_of"] = as_of
    if len(out):
        cli[args.db]["PredictedPrices"].insert_many(out.to_dict(orient="records"))


def score_point(panel, FEATURES, scaler, model, HORIZONS, HLAB, device):
    __log.info("Scoring point predictions")
    rows = []
    L = args.lookback
    for sym, g in panel.groupby(SYMBOL_COL):
        g = g.sort_values(TIME_COL)
        if len(g) < L: continue
        win = g.iloc[-L:]
        x = win[FEATURES].values.astype(np.float32)[None,:,:]; x = apply_scaler(x, scaler)
        a = np.array([int(win["asset_id"].iloc[-1])], dtype=np.int64)
        xt,at = torch.from_numpy(x).to(device), torch.from_numpy(a).to(device)
        model.eval();
        with torch.no_grad(): yh = model(xt,at).squeeze(0).cpu().numpy()
        out = {"symbol": sym}
        for h,v in zip(HORIZONS, yh):
            lab=HLAB[h]; out[f"pred_{lab}_logret"]=float(v); out[f"pred_{lab}_ret"]=float(np.expm1(v))
        rows.append(out)
    df = pd.DataFrame(rows)
    __log.info(f"Point predictions generated: rows={len(df)}")
    return df

HORIZON_DAYS = [7, 30, 182, 365]
HMAX = max(HORIZON_DAYS)

def walk_forward_ranges(min_date, max_date, train_span_days, val_span_days, step_days):
    start = pd.to_datetime(min_date)
    maxd  = pd.to_datetime(max_date)
    usable_end = maxd - pd.Timedelta(days=HMAX)   # <<< important

    train_span = pd.Timedelta(days=train_span_days)
    val_span   = pd.Timedelta(days=val_span_days)
    step       = pd.Timedelta(days=step_days)

    anchor = start + train_span
    i = 0
    while True:
        tr_start, tr_end = anchor - train_span, anchor
        va_start, va_end = anchor, anchor + val_span
        if tr_end > usable_end or va_end > usable_end:   # <<< clamp
            break
        yield i, (tr_start, tr_end), (va_start, va_end)
        i += 1
        anchor += step

class OnlineStandardizer:
    def __init__(self, n_features):
        self.n = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.M2   = np.zeros(n_features, dtype=np.float64)

    def partial_fit(self, X):  # X: [B, T, F] or [N, F]
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 3:  # collapse batch/time
            X = X.reshape(-1, X.shape[-1])
        for row in X:
            self.n += 1
            delta = row - self.mean
            self.mean += delta / self.n
            self.M2   += delta * (row - self.mean)

    def finalize(self, eps=1e-8):
        var = self.M2 / max(self.n - 1, 1)
        self.std = np.sqrt(np.maximum(var, eps)).astype(np.float32)
        self.mean = self.mean.astype(np.float32)
        return self.mean, self.std

    def transform(self, X):
        return (X - self.mean) / self.std

def safe_log(x):
    a = np.asarray(x, dtype=float)
    return np.log(np.clip(a, EPS, None))

def reduce_daily_duplicates(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values(TIME_COL).dropna(subset=[TIME_COL]).copy()
    # aggregate duplicates per day with OHLC rules
    rules = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    if g[TIME_COL].duplicated().any():
        cols = [TIME_COL] + [c for c in rules if c in g.columns]
        g = g[cols].groupby(TIME_COL, as_index=False).agg({c: rules[c] for c in cols if c != TIME_COL})
    return g

def build_sequences_one_symbol(df, lookback, features, horizons_days, label_start, label_end):
    g = df.sort_values(TIME_COL).copy()
    ts = pd.to_datetime(g[TIME_COL]).to_numpy()

    # ensure features exist & are finite
    for c in features:
        if c not in g.columns:
            g[c] = 0.0
    g[features] = (g[features].replace([np.inf, -np.inf], np.nan)
                                .ffill().bfill().fillna(0.0)).astype(np.float32)

    close = pd.to_numeric(g["close"], errors="coerce").to_numpy(np.float64)
    close = np.where((close > 0) & np.isfinite(close), close, np.nan)
    logc  = np.log(close)

    hmax = max(horizons_days) if horizons_days else 0
    L = len(g)
    if L < lookback + hmax + 1:
        return None

    sym_first = ts[0]
    sym_last = ts[-1]
    eff_start = max(pd.to_datetime(label_start), sym_first + pd.Timedelta(days=lookback - 1))
    eff_end = min(pd.to_datetime(label_end), sym_last - pd.Timedelta(days=hmax))
    if eff_end < eff_start:
        return None

    t_lo = lookback
    t_hi = L - 1 - hmax
    lbl_mask = (ts >= eff_start) & (ts <= eff_end)

    Xs, Ms, Ys = [], [], []
    for t in range(t_lo, t_hi + 1):
        if not lbl_mask[t]:  # label time outside feasible window
            continue

        win = g.iloc[t - lookback: t]
        x = win[features].to_numpy(np.float32, copy=True)
        if not np.isfinite(x).all():
            continue

        m = np.ones(lookback, dtype=bool)

        a = logc[t]
        if not np.isfinite(a):
            continue

        y = []
        ok = True
        for H in horizons_days:
            b = logc[t + H]
            if not np.isfinite(b):
                ok = False
                break
            y.append(np.float32(b - a))
        if not ok:
            continue

        Xs.append(x)
        Ms.append(m)
        Ys.append(np.asarray(y, dtype=np.float32))

    if not Xs:
        return None
    return np.stack(Xs), np.stack(Ms), np.stack(Ys)



class PanelStreamDataset(IterableDataset):

    def __init__(self, symbols, start, end, lookback, features, tcols, mcols, asset2id, horizon_days = [7, 30, 182, 365]):
        super().__init__()
        self.symbols = list(symbols)
        self.start = pd.to_datetime(start)
        self.end   = pd.to_datetime(end)
        self.lbpad = pd.Timedelta(days=lookback + 5)
        self.lookback = lookback
        self.features = features
        self.tcols = tcols
        self.mcols = mcols
        self.asset2id = asset2id
        self.horizon_days = horizon_days

    def __iter__(self):
        con = _get_duck()  # per-worker duckdb connection (as in earlier fix)
        q = """
                  SELECT * FROM read_parquet('data/PriceData.prepped.parquet')
                  WHERE symbol = ? AND date >= ? AND date <= ?
                  ORDER BY date
                """
        for sym in self.symbols:
            fpad = pd.Timedelta(days=HMAX)
            df = con.execute(q, [sym,
                                 (self.start - self.lbpad),
                                 (self.end + fpad)  # <<< pad for labels
                                 ]).df()

            #info = _diagnose_symbol(df, sym, self.lookback, self.horizon_days, self.start, self.end)
            #logging.getLogger(__name__).info("[val diag] " + info)

            if df.empty:
                continue
            df[TIME_COL] = pd.to_datetime(df[TIME_COL]).dt.normalize()

            seqs = build_sequences_one_symbol(df,
                                  lookback=self.lookback,
                                  features=self.features,
                                  horizons_days=self.horizon_days,
                                  label_start=self.start,
                                  label_end=self.end)
            if seqs is None:
                continue
            X, M, Y = seqs
            aid = int(self.asset2id.get(sym, 0))
            A = torch.tensor(aid, dtype=torch.long)  # constant per symbol
            for i in range(len(X)):
                yield (
                    sym,
                    torch.from_numpy(X[i]),  # [T, F]
                    torch.from_numpy(M[i]).bool(),  # [T]
                    A,  # scalar LongTensor
                    torch.from_numpy(Y[i])  # [H]
                )

def collate_batch(samples):
    if len(samples[0]) == 5:
        _, X, M, A, Y = zip(*samples)
    else:
        X, M, A, Y = zip(*samples)
    return (torch.stack(X),                # [B, T, F]
            torch.stack(M),                # [B, T]
            torch.stack(A),                # [B]
            torch.stack(Y))                # [B, H]

def score_streaming(model, device, con, q_panel, symbols, as_of_end,
                    lookback, features, tcols, mcols, mean, std,
                    price_idx, fx_idx, top_k=25, rank_horizon_idx=-1, asset2id=None):
    """
    Returns a list of (rank_value, symbol, extra) for the top_k symbols.
    rank_horizon_idx: which horizon to rank by (e.g., -1 = last horizon like 1y)
    """
    import heapq

    model.eval()
    heap = []
    mean_t = torch.from_numpy(mean)
    std_t  = torch.from_numpy(std)

    ds = PanelStreamDataset(
        symbols=symbols,
        start=as_of_end - pd.Timedelta(days=lookback+5),
        end=as_of_end, lookback=lookback,
        features=features, tcols=tcols, mcols=mcols,
        asset2id=asset2id
    )

    with torch.no_grad():
        for sym, X, M, A, _ in ds:
            X = ((X - mean_t) / std_t).unsqueeze(0).to(device)  # [1, T, F]
            M = M.unsqueeze(0).to(device)                       # [1, T]
            A = A.unsqueeze(0).to(device)
            price_x = X[:, :, price_idx]
            fx_x = X[:, :, fx_idx]
            weather_x = torch.zeros(price_x.size(0), price_x.size(1), 1, device=device)
            news_tokens = {
                "input_ids": torch.zeros(price_x.size(0), price_x.size(1), 1, dtype=torch.long, device=device),
                "attention_mask": torch.zeros(price_x.size(0), price_x.size(1), 1, dtype=torch.long, device=device),
            }
            pred = model(price_x, A, M, fx_x, M, weather_x, M, news_tokens, M)
            lr = float(pred[0, rank_horizon_idx].detach().cpu())
            item = {"pred_logret": lr, "as_of": str(pd.to_datetime(as_of_end).date())}
            if len(heap) < top_k:
                heapq.heappush(heap, (lr, sym, item))
            else:
                heapq.heappushpop(heap, (lr, sym, item))

    return sorted(heap, key=lambda t: t[0], reverse=True)

def _diagnose_symbol(df, sym, lookback, horizons_days, label_start, label_end):
    ts = pd.to_datetime(df[TIME_COL]).sort_values().to_numpy()
    if ts.size == 0:
        return f"{sym}: no rows"
    hmax = max(horizons_days)
    sym_first, sym_last = ts[0], ts[-1]
    eff_start = max(pd.to_datetime(label_start), sym_first + pd.Timedelta(days=lookback-1))
    eff_end   = min(pd.to_datetime(label_end),   sym_last  - pd.Timedelta(days=hmax))

    return (f"{sym}: rows={len(ts)} "
            f"sym_range=[{str(sym_first)},{str(sym_last)}], "
            f"eff_lbl=[{str(eff_start.date())},{str(eff_end.date())}]")


# B) Second pass: training with scaling applied on the fly
class ScaledWrapper(IterableDataset):
    def __init__(self, base, mean, std):
        self.base = base
        self.mean = torch.from_numpy(mean)
        self.std  = torch.from_numpy(std)

    def __iter__(self):
        for S, X, M, A, Y in self.base:
            yield (S, (X - self.mean) / self.std, M, A, Y)

def train():
    con = duckdb.connect()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    q_sym = f"""
              SELECT DISTINCT symbol
              FROM read_parquet('data/PriceData.parquet')
              ORDER BY symbol
            """

    symbols = con.execute(q_sym).df()["symbol"].dropna().tolist()
    asset2id = {s: i for i, s in enumerate(sorted(symbols))}

    min_date, max_date = con.execute("SELECT min(date), max(date) FROM read_parquet('data/PriceData.prepped.parquet')").fetchone()

    q_panel = f"""
            SELECT *
            FROM read_parquet('data/PriceData.prepped.parquet')
            ORDER BY symbol, date ASC
            """

    q_file = "read_parquet('data/PriceData.prepped.parquet')"

    panel = con.execute(q_panel).df()

    cal = [int(x) for x in args.horizons_cal.split(",") if x]
    HLAB = {7: "1w", 30: "1m", 182: "6m", 365: "1y"}
    tcols, mcols = build_target_and_mask(cal)

    features = [c for c in panel.columns.tolist() if c not in ["symbol", "date", "source"] and not c.startswith(("target_", "mask_"))]
    fx_features = ["eur_fx_logret"] if "eur_fx_logret" in features else []
    price_features = [c for c in features if c not in fx_features]
    price_idx = [features.index(c) for c in price_features]
    fx_idx = [features.index(c) for c in fx_features]

    model = None
    scaler = None
    H = len(cal)

    for fold, (i, tr_rng, va_rng) in enumerate(walk_forward_ranges(min_date, max_date,
                                                                args.train_span_days, args.val_span_days,
                                                                args.step_days), 1):
        tr_start, tr_end = tr_rng
        va_start, va_end = va_rng
        __log.info(f"[fold {fold}] train {tr_start.date()}→{tr_end.date()}  val {va_start.date()}→{va_end.date()}")
        #__log.info(
        #    f"[fold {fold}] train {tr_start.date()}→{tr_end.date()}  val {va_start.date()}→{va_end.date()}  (usable_end={(pd.to_datetime(max_date) - pd.Timedelta(days=max(HORIZON_DAYS))).date()})")

        # make sure this line always runs inside the fold, before the loop
        scaler = OnlineStandardizer(n_features=len(features))

        seen = False
        ds_tr_pass1 = PanelStreamDataset(symbols, tr_start, tr_end, args.lookback,
                                         features, tcols, mcols, asset2id)

        for X, M, A, Y in DataLoader(ds_tr_pass1, batch_size=1, num_workers=0, collate_fn=collate_batch):
            scaler.partial_fit(X.numpy())
            seen = True

        if not seen:
            __log.warning(f"No training samples in {tr_start.date()}→{tr_end.date()} — skipping fold.")
            continue

        mean, std = scaler.finalize()

        # Training datasets/loaders
        base_tr = PanelStreamDataset(symbols, tr_start, tr_end, args.lookback, features, tcols, mcols, asset2id)
        ds_tr = ScaledWrapper(base_tr, mean, std)
        dl_tr = DataLoader(ds_tr, batch_size=args.batch, num_workers=args.num_workers,
                   worker_init_fn=worker_init_fn, collate_fn=collate_batch, pin_memory=True, persistent_workers=True)

        base_va = PanelStreamDataset(symbols, va_start, va_end, args.lookback, features, tcols, mcols, asset2id)
        ds_va = ScaledWrapper(base_va, mean, std)
        dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False,
                           num_workers=args.num_workers, worker_init_fn=worker_init_fn,
                           collate_fn=collate_batch, pin_memory=True, persistent_workers=True)

        # C) Build model & train using multimodal encoders
        H = len(cal)  # e.g., 4 for [7,30,182,365]
        price_enc = PriceEncoder(
            n_features=len(price_features),
            n_assets=len(symbols),
            d_model=args.hidden,
            layers=args.layers,
            dropout=args.dropout,
        )
        fx_enc = FXEncoder(n_features=len(fx_features), d_model=args.hidden)
        weather_enc = ZeroWeatherEncoder(args.hidden)
        news_enc = ZeroNewsEncoder(args.hidden)
        model = MultiModalPriceForecast(price_enc, fx_enc, weather_enc, news_enc,
                                        out_dim=H, hidden=args.hidden, dropout=args.dropout)
        n_val = 0
        for _ in dl_va:
            n_val += 1
        __log.info(f"val batches: {n_val}")

        probe = PanelStreamDataset(symbols, va_start, va_end, args.lookback,
                                   features, tcols, mcols, asset2id,
                                   horizon_days=HORIZON_DAYS)
        val_count = sum(1 for _ in DataLoader(probe, batch_size=1, num_workers=0, collate_fn=collate_batch))
        __log.info(f"[debug] val candidate sequences (batch=1): {val_count}")

        model, best = train_point(model, dl_tr, dl_va, device, args.epochs, args.lr, args.weight_decay,
                                  price_idx=price_idx, fx_idx=fx_idx)
        __log.info(f"[fold {fold}] best val: {best:.6f}")

        del dl_tr, dl_va
        gc.collect()

    if scaler is None:
        return

    trained_mean, trained_std = scaler.finalize()

    as_of_end = pd.to_datetime("today").normalize()
    rank_h_idx = len(cal) - 1
    mean, std = trained_mean, trained_std

    top = score_streaming(
        model=model, device=device, con=con, q_panel=q_file,
        symbols=symbols, as_of_end=as_of_end, lookback=args.lookback,
        features=features, tcols=tcols, mcols=mcols,
        mean=mean, std=std, price_idx=price_idx, fx_idx=fx_idx,
        top_k=25, rank_horizon_idx=rank_h_idx,
        asset2id=asset2id  # pass None if your model forward is (X, M) only
    )

    # Convert to a DataFrame you can persist
    pred_df = pd.DataFrame(
        [{"symbol": s, "pred_1y_logret": lr, "as_of": item["as_of"]} for lr, s, item in top]
    )
    persist_topk_to_mongo(pred_df, 25)

def main():
    __log.info("Starting up training step")
    set_seed(args.seed)
    train()


if __name__ == "__main__":
    from util.argument_parser import parse_args
    args = parse_args()
    setup_diagnostics()
    main()
    disable_diagnostics()
