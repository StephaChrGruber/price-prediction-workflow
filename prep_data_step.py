import gc
import logging
import os
import sys
from argparse import Namespace
from typing import List

import numpy as np
from pyarrow import Schema
from pyarrow.parquet import ParquetFile, ParquetWriter
from urllib3.filepost import writer

from util.data_utils import stream_parquet, align_df_to_schema
from util.diagnostics import set_seed, setup_diagnostics, disable_diagnostics, safe_log
import polars as pl
from util.constants import TIME_COL, SYMBOL_COL, EPS
import pyarrow.parquet as pq
import pyarrow as pa
import duckdb
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)-25s] %(asctime)s [%(levelname)-8s]: %(message)s",
    stream=sys.stdout,
    force=True
)

__log = logging.getLogger(__name__)

args: Namespace

def to_dt(s: pl.Series) -> pl.Series:
    return s.dt.replace_time_zone(None).dt.replace(hour=0,minute=0,second=0,microsecond=0)

def prep_crypt(df: pl.DataFrame):
    __log.info("Preparing crypto")
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

    d = d.with_columns([to_dt(pl.col(TIME_COL))])
    num_cols = [c for c in needed if c not in {TIME_COL, SYMBOL_COL}]
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
    __log.info(f"Prepared crypto: shape={out.shape}")
    return out

def prep_stocks(df: pl.DataFrame) -> pl.DataFrame:
    __log.info("Preparing socks")
    if df.is_empty():
        return df

    d = df.clone()
    needed = ["date", "symbol", "open", "close", "high", "low", "volume"]
    for k in needed:
        if k not in d.columns:
            d = d.with_columns(pl.lit(np.nan).alias(k))

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

    __log.info(f"Prepared stocks: shape={out.shape}")
    return out

def prep_fx(df: pl.DataFrame) -> pl.DataFrame:
    __log.info("Preparing FX")
    if df.is_empty():
        return df

    d = df.clone()
    d = d.with_columns(d.get_column(TIME_COL))
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

    __log.info(f"Prepared FX: shape={out.shape}")

    return out

def merge_parquet(first_in_path: str, second_in_path: str, out_path: str,chunk_rows: int = 100_000,compression: str = "zstd", del_in: bool = True):
    __log.info(f"Merging {first_in_path} and {second_in_path} into {out_path}")
    if os.path.exists(out_path):
        os.remove(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pf_first = pq.ParquetFile(first_in_path)
    pf_second = pq.ParquetFile(second_in_path)

    first_schema: Schema = pf_first.schema_arrow
    second_schema: Schema = pf_second.schema_arrow
    schema = []

    for s in first_schema:
        schema.append((s.name, s.type))

    for s in second_schema:
        if s.name not in second_schema.names:
            schema.append((s.name, s.type))

    schema = pa.schema(schema)

    def _copy_to_merged(pf_in: ParquetFile, writer: ParquetWriter):
        for i, batch in enumerate(pf_in.iter_batches(batch_size=chunk_rows)):
            df = pl.from_arrow(batch)
            df = align_df_to_schema(df, schema)
            tbl = df.to_arrow()

            writer.write_table(tbl)
            del df, tbl, batch
            gc.collect()

    pq_writer = pq.ParquetWriter(out_path, schema, compression=compression)

    _copy_to_merged(pf_first, pq_writer)
    _copy_to_merged(pf_second, pq_writer)

    pq_writer.close()

    if del_in:
        os.remove(first_in_path)
        os.remove(second_in_path)

def daily_calendar(file: str):
    __log.info("Generating Daily Calendar")
    first = (pl.scan_parquet(file)
           .with_columns(pl.col(TIME_COL).alias(TIME_COL))  # if string dates
           .select(pl.col(TIME_COL).max().alias("max_date"))
           .collect(streaming=True))["max_date"][0]
    last = (pl.scan_parquet(file)
             .with_columns(pl.col(TIME_COL).alias(TIME_COL))  # if string dates
             .select(pl.col(TIME_COL).min().alias("min_date"))
             .collect(streaming=True))["min_date"][0]

    return pd.date_range(first, last, freq='D')

def iter_groups_ordered_sym_date(path_glob: str, symbols, batch_rows: int = 200_000):
    con = duckdb.connect()
    q = f"""
      SELECT *
      FROM read_parquet('{path_glob}')
      GROUP BY symbol, date
      ORDER BY symbol, date
    """
    pending_key = None
    pending_df: pd.DataFrame | None = None

    for rb in con.execute(q).fetch_record_batch(batch_rows):
        df: pd.DataFrame = pl.from_arrow(rb).to_pandas() # columns: symbol, date, ...

        # rows are sorted, so groups are contiguous
        for key, g in df.groupby(["symbol","date"]):
            if pending_key is None:
                pending_key, pending_df = key, g
            elif key == pending_key:
                pending_df = pd.concat([pending_df, g], axis="rows")
            else:
                yield pending_key, pending_df
                pending_key, pending_df = key, g

    if pending_df is not None:
        yield pending_key, pending_df

def prep_one_symbol(sym: str, g: pd.DataFrame, asset2id, cal, horizon_days: List[int]) -> pd.DataFrame:
    __log.info(f"Processing: {sym}")
    try:
        # keep source before reindex drops it
        src = g["source"].iloc[0] if "source" in g.columns and len(g) else "unknown"

        # 1) align to full daily calendar
        g = g.set_index(TIME_COL)

        # 2) mark trading days
        g["is_trading_day"] = g["close"].notna()

        # 3) fill & clamp prices (safe)
        for col in ["close", "open", "high", "low"]:
            g[col] = pd.to_numeric(g.get(col, np.nan), errors="coerce").ffill().bfill().fillna(0.0).clip(lower=EPS)

        # 4) activity features (0 on non-trading days)
        for col in ["volume", "quote_asset_volume", "num_trades", "taker_buy_base_vol", "taker_buy_quote_vol",
                    "closePctChange", "openPctChange"]:
            g[col] = pd.to_numeric(g.get(col, 0.0), errors="coerce").fillna(0.0)
        for col in ["volume", "quote_asset_volume", "num_trades", "taker_buy_base_vol", "taker_buy_quote_vol"]:
            g[col] = g[col] * g["is_trading_day"].fillna(0)

        logc = np.log(g["close"]).diff(1)
        g["logret_1d"] = np.log(g["close"]).diff(1)
        g["logret_7d"] = np.log(g["close"]).diff(5)
        g["logret_30d"] = np.log(g["close"]).diff(30)
        g["vol_1d"] = pd.Series(g["logret_1d"], index=g.index).rolling(1, min_periods=1).std().fillna(0.0).to_numpy()
        g["vol_7d"] = pd.Series(g["logret_1d"], index=g.index).rolling(7, min_periods=1).std().fillna(0.0).to_numpy()
        g["vol_30d"] = pd.Series(g["logret_1d"], index=g.index).rolling(30, min_periods=1).std().fillna(0.0).to_numpy()
        g["mom_1d"] = logc - np.roll(logc, 1)
        g.iloc[:1, g.columns.get_loc("mom_1d")] = 0.0
        g["mom_7d"] = logc - np.roll(logc, 7)
        g.iloc[:7, g.columns.get_loc("mom_7d")] = 0.0
        g["mom_30d"] = logc - np.roll(logc, 30)
        g.iloc[:30, g.columns.get_loc("mom_30d")] = 0.0
        g["hl_spread"] = ((g["high"] - g["low"]) / np.maximum(g["close"], EPS)).astype(float)

        # 6) restore ids
        g[SYMBOL_COL] = sym
        g[TIME_COL] = g.index
        g["source"] = src

        g["dow"] = g[TIME_COL].dt.dayofweek
        g["dom"] = g[TIME_COL].dt.day
        g["month"] = g[TIME_COL].dt.month

        g["asset_id"] = g[SYMBOL_COL].map(asset2id).astype(np.int32)

        for h in horizon_days:
            tcol, mcol = f"target_{h}d", f"mask_{h}d"
            g[tcol] = g["close"].shift(-h) - g["close"]
            g[mcol] = np.isfinite(g[tcol]).astype("int8")

        return g.reset_index(drop=True)

    except Exception as e:
        # If you have a cross-process safe logger, use it; else print
        __log.exception(f"[worker] symbol={sym} failed: {e}")
        # Re-raise so the main process can also log the failure and skip the symbol
        raise

def pre_prices():
    cal = daily_calendar("data/PriceData.parquet")

    os.makedirs("/tmp/duck_spill", exist_ok=True)
    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA memory_limit='4GB';")  # cap RAM
    con.execute("PRAGMA temp_directory='/tmp/duck_spill';")

    writer = None

    q_sym = f"""
              SELECT DISTINCT symbol
              FROM read_parquet('data/PriceData.parquet')
              ORDER BY symbol
            """

    symbols = con.execute(q_sym).df()
    asset2id = {s: i for i, s in enumerate(sorted(symbols["symbol"].dropna()))}

    cal = [int(x) for x in args.horizons_cal.split(",") if x]

    HORIZONS = cal
    HLAB = {7: "1w", 30: "1m", 182: "6m", 365: "1y"}

    for _, sym in symbols.iterrows():
        symbol = sym.tolist()[0]
        q_one_sym = f"""
                  SELECT *
                  FROM read_parquet('data/PriceData.parquet')
                  WHERE symbol = '{symbol}'
                  ORDER BY date ASC
                """
        data = con.execute(q_one_sym).df()
        data = pl.from_pandas(prep_one_symbol(symbol, data,asset2id, cal, horizon_days=HORIZONS)).to_arrow()

        if writer is None:
            writer = pq.ParquetWriter("data/PriceData.prepped.parquet", data.schema)

        writer.write_table(data)

    writer.close()


def prep_data():

    stream_parquet(
        "data/DailyCryptoData.parquet",
        "data/DailyCryptoData.prepped.parquet",
        prep_crypt,
        chunk_rows=100_000
    )
    stream_parquet(
        "data/DailyStockData.parquet",
        "data/DailyStockData.prepped.parquet",
        prep_stocks,
        chunk_rows=100_000
    )

    merge_parquet("data/DailyCryptoData.prepped.parquet",
                  "data/DailyStockData.prepped.parquet",
                  "data/PriceData.parquet")


    pre_prices()

    stream_parquet(
        "data/DailyFxData.parquet",
        "data/DailyFxData.prepped.parquet",
        prep_fx,
        chunk_rows=500
    )




def main():
    __log.info("Starting up data prep step")
    set_seed(args.seed)
    prep_data()


if __name__ == "__main__":
    from util.argument_parser import parse_args
    args = parse_args()
    setup_diagnostics()
    main()
    disable_diagnostics()