import json
import logging
import os
import pickle
import sys
from argparse import Namespace
from pathlib import Path
from typing import Optional, Mapping, Any, Sequence
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

from training_step import PanelStreamDataset
from util.diagnostics import set_seed, setup_diagnostics, disable_diagnostics

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)-25s] %(asctime)s [%(levelname)-8s]: %(message)s",
    stream=sys.stdout,
    force=True
)

__log = logging.getLogger(__name__)

args: Namespace

ART = Path("artifacts")
MODEL_PATH     = ART / "model.pt"
SCALER_PATH    = ART / "scaler.json"
CALENDAR_PATH  = ART / "calendar.csv"
SYMBOLS_PATH   = ART / "symbols.json"
FEATURES_PATH  = ART / "features.json"
TCOLS_PATH     = ART / "tcols.json"
MCOLS_PATH     = ART / "mcols.json"
ASSET2ID_PATH  = ART / "asset2id.json"     # optional
METADATA_PATH  = ART / "metadata.json"
OUTPUT_CSV     = ART / "predictions.csv"

def _load_json_list(p: Path) -> list[str]:
    return list(map(str, json.loads(p.read_text())))

def _load_mean_std(p: Path) -> tuple[Sequence[float], Sequence[float]]:
    data = json.loads(p.read_text())
    return data["mean"], data["std"]

def _maybe_load_asset2id(p: Path) -> Optional[Mapping[str, int]]:
    return json.loads(p.read_text()) if p.exists() else None

def _build_con_from_meta(meta: Mapping[str, Any]) -> Any:
    # Try duckdb first if provided
    duckdb_path = meta.get("duckdb_path")
    if duckdb_path:
        import duckdb
        return duckdb.connect(duckdb_path)

    # Try SQLAlchemy URL if provided
    db_url = meta.get("db_url")
    if db_url:
        from sqlalchemy import create_engine
        return create_engine(db_url).connect()

    # Otherwise, let score_streaming rely on q_panel file or internal logic
    return None

def _maybe_persist_to_mongo(df: pd.DataFrame, meta: Mapping[str, Any], top_k: int) -> None:
    uri = meta.get("mongo_uri")
    db  = meta.get("mongo_db")
    coll= meta.get("mongo_coll")
    if not (uri and db and coll):
        return
    from pymongo import MongoClient
    cli = MongoClient(uri)
    col = cli[db][coll]
    docs = df.sort_values("pred_1y_logret", ascending=False).head(top_k).to_dict(orient="records")
    if docs:
        col.insert_many(docs)

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
    mean_t = torch.from_numpy(np.array(mean))
    std_t  = torch.from_numpy(np.array(std))

    ds = PanelStreamDataset(
        symbols=symbols,
        start=as_of_end,
        end=as_of_end,
        lookback=lookback,
        features=features,
        tcols=tcols,
        mcols=mcols,
        asset2id=asset2id,
        horizon_days=[],  # inference should not require future label horizons
        allow_label_fallback=True,
    )

    produced = False
    with torch.no_grad():
        for sym, X, M, A, _ in tqdm(ds, desc="Iteration"):
            produced = True
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

    if not produced:
        __log.warning("PanelStreamDataset produced no sequences; check input data and parameters")

    return sorted(heap, key=lambda t: t[0], reverse=True)

def predict():
    # 0) Load metadata & constants
    meta = json.loads(METADATA_PATH.read_text())
    lookback = int(meta.get("lookback"))
    price_idx = list(meta.get("price_idx"))
    fx_idx = meta.get("fx_idx", None)
    fx_idx = None if fx_idx is None else list(fx_idx)
    q_panel = meta.get("q_panel", None)
    top_k = int(meta.get("top_k", 25))

    # 1) Load model and artifacts
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model at {MODEL_PATH}")
    model = torch.load(MODEL_PATH.as_posix(), map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 2) Load the checkpoint
    ckpt = torch.load(ART / "model_state.pt", map_location=device)  # safe if file is trusted
    state = ckpt.get("model", ckpt)  # if it's a raw state_dict, this still works

    # 4) Apply it to the model
    missing, unexpected = model.load_state_dict(state, strict=False)  # strict=True if you want hard fail
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model.to(device).eval()

    mean, std = _load_mean_std(SCALER_PATH)
    symbols = _load_json_list(SYMBOLS_PATH)
    features = _load_json_list(FEATURES_PATH)
    tcols = _load_json_list(TCOLS_PATH)
    mcols = _load_json_list(MCOLS_PATH)
    asset2id = _maybe_load_asset2id(ASSET2ID_PATH)

    cal_df = pd.read_csv(CALENDAR_PATH)
    if "date" not in cal_df.columns:
        raise ValueError("calendar.csv must contain a 'date' column")
    cal = pd.to_datetime(cal_df["date"]).sort_values(ignore_index=True)
    rank_h_idx = len(cal) - 1

    as_of_end = pd.Timestamp.today().normalize()
    ignore_days = getattr(args, "ignore_days", 0) or 0
    if ignore_days > 0:
        as_of_end = as_of_end - pd.Timedelta(days=ignore_days)

    earliest_calendar = pd.to_datetime(cal.iloc[0]).normalize()
    if as_of_end < earliest_calendar:
        as_of_end = earliest_calendar

    __log.info("Using %s as evaluation cutoff", as_of_end.date())

    # 2) Connection (optional) built from metadata
    con = _build_con_from_meta(meta)

    # 3) Score
    top = score_streaming(
        model=model,
        device=device,
        con=con,
        q_panel=q_panel,  # may be None if score_streaming doesn't need it
        symbols=symbols,
        as_of_end=as_of_end,
        lookback=lookback,
        features=features,
        tcols=tcols,
        mcols=mcols,
        mean=mean,
        std=std,
        price_idx=price_idx,
        fx_idx=fx_idx,
        top_k=top_k,
        rank_horizon_idx=rank_h_idx,
        asset2id=asset2id,
    )

    # 4) Format + save
    pred_df = pd.DataFrame(
        [{"symbol": s, "pred_1y_logret": lr, "as_of": item["as_of"]} for lr, s, item in top]
    )
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(OUTPUT_CSV, index=False)

    # 5) Optional persistence to Mongo (only if metadata contains settings)
    _maybe_persist_to_mongo(pred_df, meta, top_k)

    print(f"Wrote {len(pred_df)} predictions to {OUTPUT_CSV}")


def main():
    __log.info("Starting up prediction step")
    set_seed(args.seed)
    predict()


if __name__ == "__main__":
    from util.argument_parser import parse_args
    args = parse_args()
    setup_diagnostics()
    main()
    disable_diagnostics()
