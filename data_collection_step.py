import logging
import sys
from argparse import Namespace
from datetime import datetime, timedelta

from pymongo.collection import Collection
from pymongo.database import Database

from util.data_utils import iter_mongo_df_chunks, stage_collection_with_schema, mongo_client

from util.diagnostics import set_seed, setup_diagnostics, disable_diagnostics
import pyarrow as pa
import numpy as np
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)-25s] %(asctime)s [%(levelname)-8s]: %(message)s",
    stream=sys.stdout,
    force=True
)

__log = logging.getLogger(__name__)

__db: Database

args: Namespace

def sanitize_list_column(series: pl.Series, dtype: np.dtype, fixed_len: int | None) -> pl.Series:
    __log.debug(f"[sanitize] dtype={dtype} fixed_len={fixed_len}")
    def _clean(val):
        try:
            arr = np.asarray(val, dtype=dtype)
            if fixed_len is not None:
                arr = arr[:fixed_len]
                if arr.size < fixed_len:
                    arr = np.pad(arr, (0, fixed_len - arr.size))
            return arr.tolist()
        except Exception:
            return [0.0] * (fixed_len or 0)

    return _clean(series)

# ---- SCHEMA ----

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

def news_schema() -> pa.Schema:
    return pa.schema([
        pa.field("date", pa.timestamp("ns")),
        pa.field("headline", pa.string()),
        # list<float32> (variable length)
        pa.field("sentiment", pa.list_(pa.float32())),
        # fixed_size_list<float32, 512> (exactly 512 elements per row)
        pa.field("embeddings", pa.list_(pa.float32(), 512)),
    ])

# ---- ITER FUNC ----

def iter_crypto_chunks(start_id=None):
    for df, last_id in iter_mongo_df_chunks(
            __db.DailyCryptoData,
            query={"date": {
                "$lte": datetime.today().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(
                    days=args.ignore_days)}},
            chunk_rows=100_000,
            start_id=start_id
    ):
        yield df, last_id

def iter_stocks_chunks(start_id=None):
    for df, last_id in iter_mongo_df_chunks(
            __db.DailyStockData,
            query={"date" : {"$lte" : datetime.today().replace(hour = 0, minute = 0, second=0, microsecond=0) - timedelta(days=args.ignore_days)}},
            projection={"date": 1, "symbol": 1, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
            chunk_rows=100_000,
            start_id=start_id
    ):
            yield df, last_id

def iter_weather_chunks(start_id=None):
    for df, last_id in iter_mongo_df_chunks(
            __db.DailyWeatherData,
            query={"date": {
                "$lte": datetime.today().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=args.ignore_days)}},
            chunk_rows=100_000,
            start_id=start_id
    ):
        yield df, last_id

def iter_fx_chunks(start_id=None):
    for df, last_id in iter_mongo_df_chunks(
            __db.DailyCurrencyExchangeRates,
            query={"date": {
                "$lte": datetime.today().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=args.ignore_days)}},
            chunk_rows=200_000,
            start_id=start_id
    ):
        yield df, last_id

def iter_news_chunks(start_id=None):
    for df, last_id in iter_mongo_df_chunks(
            __db.NewsHeadlines,
            query={"date": {
                "$lte": datetime.today().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=args.ignore_days)}},
            projection={"date": 1, "headline": 1, "sentiment": 1, "embeddings": 1},
            chunk_rows=20000,
            start_id=start_id
    ):
        # sanitize arrays
        df = df.with_columns(
            sanitize_list_column(df.get_column("sentiment"), np.float32, fixed_len=None)
        )
        yield df, last_id

def pull_data_from_mongo():
    global __db
    __db = mongo_client()[args.db]

    stage_collection_with_schema(
        out_path="data/DailyCryptoData.parquet",
        iter_chunks_fn=iter_crypto_chunks,
        canonical_schema=crypto_schema(32),
        resume=True,
        compression="zstd"
    )

    stage_collection_with_schema(
        out_path="data/DailyStockData.parquet",
        iter_chunks_fn=iter_stocks_chunks,
        canonical_schema=stock_schema(32),
        resume=True,
        compression="zstd"
    )

    stage_collection_with_schema(
        out_path="data/DailyFxData.parquet",
        iter_chunks_fn=iter_fx_chunks,
        canonical_schema=fx_schema(32),
        resume=True,
        compression="zstd"
    )

    """stage_collection_with_schema(
        out_path="data/DailyWeatherData.parquet",
        iter_chunks_fn=iter_weather_chunks,
        canonical_schema=weather_schema(32),
        resume=True,
        compression="zstd"
    )

    stage_collection_with_schema(
        out_path="data/DailyNewsData.parquet",
        iter_chunks_fn=iter_news_chunks,
        canonical_schema=news_schema(),
        resume=True,
        compression="zstd"
    )"""

def main():
    __log.info("Starting up data collection step")
    set_seed(args.seed)

    pull_data_from_mongo()


if __name__ == "__main__":
    from util.argument_parser import parse_args
    args = parse_args()
    setup_diagnostics()
    main()
    disable_diagnostics()