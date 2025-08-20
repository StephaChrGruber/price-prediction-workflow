import logging
import os
from typing import Iterator, Optional, Dict, Any, Tuple

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from bson import ObjectId
from pymongo import MongoClient

from diagnostics import log_mem

logger = logging.getLogger(__name__)

# -------------------------
# Mongo helpers
# -------------------------

def mongo_client() -> MongoClient:
    uri = (
        f"mongodb://admin:2059$tephan5203@94.130.171.244:27017,host.docker.internal:27017,homeassistant.local:27017"
        f"/PriceForecast?authSource=admin&replicaSet=rs0&readPreference=primary&w=majority&retryWrites=true"
    )

    return MongoClient(
        uri,
        compressors="zstd,snappy",
        serverSelectionTimeoutMS=20_000,
        socketTimeoutMS=600_000,
        connectTimeoutMS=20_000,
        maxPoolSize=8,
    )


def iter_mongo_df_chunks(
    coll,
    query: Optional[Dict[str, Any]] = None,
    projection: Optional[Dict[str, int]] = None,
    chunk_rows: int = 200_000,
    batch_size_db: int = 5_000,
    start_id: Optional[str] = None,
) -> Iterator[Tuple[pl.DataFrame, str]]:
    query = dict(query or {})
    projection = projection
    last_id = ObjectId(start_id) if start_id else None

    while True:
        q = query.copy()
        if last_id:
            q["_id"] = {"$gt": last_id}

        cur = (
            coll.find(q, projection)
            .sort([("_id", 1)])
            .limit(chunk_rows)
            .batch_size(batch_size_db)
        )

        docs = list(cur)
        cur.close()
        if not docs:
            break

        last_id = docs[-1]["_id"]
        for d in docs:
            d.pop("_id", None)

        df = pl.DataFrame(docs, infer_schema_length=None,   # scan all rows, not just first 50
    nan_to_null=True,           # turn NaN -> null on ingest
    strict=False,).fill_null(0.0)
        yield df, str(last_id)


# -------------------------
# Parquet utilities
# -------------------------

def _arrow_field_to_pl_dtype(field: pa.Field):
    t = field.type
    if pa.types.is_timestamp(t):
        return pl.Datetime
    if pa.types.is_string(t):
        return pl.Utf8
    if pa.types.is_float32(t):
        return pl.Float32
    if pa.types.is_float64(t):
        return pl.Float64
    if pa.types.is_int32(t):
        return pl.Int32
    if pa.types.is_int64(t):
        return pl.Int64
    return None


def align_df_to_schema(df: pl.DataFrame, schema: pa.Schema) -> pl.DataFrame:
    out = df.clone()
    for field in schema:
        name = field.name
        if name not in out.columns:
            pl_type = _arrow_field_to_pl_dtype(field)
            out = out.with_columns(pl.lit(None).cast(pl_type or pl.Null).alias(name))
        pl_type = _arrow_field_to_pl_dtype(field)
        if pl_type is None:
            continue
        if pl_type == pl.Datetime:
            out = out.with_columns(pl.col(name).dt.replace_time_zone(None))
        elif pl_type in (pl.Float32, pl.Float64):
            out = out.with_columns(pl.col(name).cast(pl_type, strict=False))
        elif pl_type in (pl.Int32, pl.Int64):
            out = out.with_columns(pl.col(name).cast(pl_type, strict=False).fill_null(0))
        elif pl_type == pl.Utf8:
            out = out.with_columns(pl.col(name).cast(pl.Utf8, strict=False))
    out = out.select([f.name for f in schema])
    return out


async def stage_collection_with_schema(
    out_path: str,
    iter_chunks_fn,
    canonical_schema: pa.Schema | None = None,
    resume: bool = True,
    compression: str = "zstd",
):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ckpt = out_path + ".ckpt"

    target_schema = None
    append = False
    if os.path.exists(out_path):
        try:
            target_schema = pq.ParquetFile(out_path).schema_arrow
            append = True
            logger.info(f"[stage] Using existing file schema: {target_schema}")
        except Exception as e:
            logger.exception(f"[stage] Cannot read existing schema: {e}")
            append = True
    if target_schema is None:
        target_schema = canonical_schema

    start_id = None
    if resume and os.path.exists(ckpt):
        start_id = (open(ckpt).read().strip() or None)
    writer = None
    for i, (df, last_id) in enumerate(iter_chunks_fn(start_id)):
        if target_schema is None:
            df = align_df_to_schema(df, canonical_schema or df.schema)
            table = df.to_arrow()
            target_schema = table.schema
        else:
            df = align_df_to_schema(df, target_schema)
            table = pa.Table.from_arrays([df[c].to_arrow() for c in target_schema.names], schema=target_schema)

        if writer is None:
            writer = pq.ParquetWriter(out_path, target_schema, compression=compression)
        writer.write_table(table)
        append = True
        with open(ckpt, "w") as f:
            f.write(last_id)
        logger.info(f"[stage] wrote chunk {i}, rows={len(df)}, checkpoint={last_id}")


def stage_collection(coll, out_path, query=None, projection=None, chunk_rows=200_000):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ckpt = out_path + ".ckpt"

    start_id = None
    if os.path.exists(ckpt):
        start_id = open(ckpt).read().strip() or None

    append = os.path.exists(out_path)
    for i, (df, last_id) in enumerate(
        iter_mongo_df_chunks(coll, query=query, projection=projection, chunk_rows=chunk_rows, start_id=start_id)
    ):
        log_mem(f"chunk {i} before write")
        table = df.to_arrow()
        pq.write_table(table, out_path, compression="zstd", append=append)
        append = True
        with open(ckpt, "w") as f:
            f.write(last_id)
        logger.info(f"[stage] wrote chunk {i}, rows={len(df)}")
        log_mem(f"chunk {i} after write")


# -------------------------
# Misc helpers
# -------------------------

def sanitize_list_column(series: pl.Series, dtype: np.dtype, fixed_len: int | None) -> pl.Series:
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

    pl_type = pl.List(pl.Float32 if dtype == np.float32 else pl.Float64)
    return series.map_elements(_clean).cast(pl_type)


def table_from_df_and_schema(df: pl.DataFrame, schema: pa.Schema) -> pa.Table:
    df = align_df_to_schema(df, schema)
    arrays = [df[name].to_arrow() for name in schema.names]
    return pa.Table.from_arrays(arrays, schema=schema)
