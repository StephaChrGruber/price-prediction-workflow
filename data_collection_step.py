import logging
import sys
from argparse import Namespace
from datetime import datetime, timedelta

from pymongo.collection import Collection
from pymongo.database import Database

from util.data_utils import iter_mongo_df_chunks, stage_collection_with_schema, mongo_client

from util.diagnostics import set_seed, setup_diagnostics, disable_diagnostics
import pyarrow as pa

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)-25s] %(asctime)s [%(levelname)-8s]: %(message)s",
    stream=sys.stdout,
    force=True
)

__log = logging.getLogger(__name__)

__db: Database

args: Namespace

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