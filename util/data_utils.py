import gc
import logging
import os
from typing import Optional, Dict, Any, Iterator, Tuple

from bson import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

__log = logging.getLogger(__name__)

def mongo_client() -> MongoClient:
    """
    Creates the MongoDB client
    :return: a client
    """
    uri = (
        f"mongodb://admin:2059$tephan5203@94.130.171.244:27017,host.docker.internal:27017,homeassistant.local:27017"
        f"/PriceForecast?authSource=admin&replicaSet=rs0&readPreference=primary&w=majority&retryWrites=true"
    )
    __log.info("Creating MongoClient")
    return MongoClient(
        uri,
        compressors="zstd,snappy",
        serverSelectionTimeoutMS=20_000,
        socketTimeoutMS=600_000,
        connectTimeoutMS=20_000,
        maxPoolSize=8,
    )

def iter_mongo_df_chunks(coll: Collection,
                          query: Optional[Dict[str, Any]] = None,
                          projection: Optional[Dict[str, any]] = None,
                          chunk_rows: int = 200_000,
                          batch_size_db: int = 5_000,
                          start_id: Optional[str] = None
                        ) -> Iterator[Tuple[pl.DataFrame, str]]:
    query = dict(query or {})
    last_id = ObjectId(start_id) if start_id else None
    __log.info(f"Start iterating chunks chunk_rows={chunk_rows} start_id={start_id}")
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
            __log.info("No more documents, stopping iteration")
            break

        last_id = docs[-1]["_id"]
        for d in docs:
            d.pop("_id", None)

        df = (
            pl.DataFrame(
                docs,
                infer_schema_length=None,
                nan_to_null=True,
                strict=True
            )
            .fill_null(0.0)
            .shrink_to_fit()
        )

        __log.info(f"Yielding chunk rows={len(df)} last_id={last_id}")
        yield df, str(last_id)
        del df
        del docs
        gc.collect()

def arrow_field_to_pl_dtype(field: pa.Field):
    t = field.type
    if pa.types.is_timestamp(t):
        return pl.Datetime
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return pl.Utf8
    if pa.types.is_float32(t):
        return pl.Float32
    if pa.types.is_float64(t):
        return pl.Float64
    if pa.types.is_int8(t):
        return pl.Int8
    if pa.types.is_int16(t):
        return pl.Int16
    if pa.types.is_int32(t):
        return pl.Int32
    if pa.types.is_int64(t):
        return pl.Int64
    if pa.types.is_boolean(t):
        return pl.Boolean
    return None

def align_df_to_schema(df: pl.DataFrame, schema: pa.Schema) -> pl.DataFrame:
    __log.info(f"Aligning df shape={df.shape} to schema")
    out = df.clone()
    for field in schema:
        name = field.name

        if name not in out.columns:
            pl_type = arrow_field_to_pl_dtype(field)
            out = out.with_columns(pl.lit(None).cast(pl_type or pl.Null).alias(name))


        pl_type = arrow_field_to_pl_dtype(field)
        if pl_type is None:
            continue

        if pl_type == pl.Datetime:
            out = out.with_columns(pl.col(name).dt.replace_time_zone(None).dt.replace(hour=0,minute=0,second=0,microsecond=0))
        elif pl_type in (pl.Float32, pl.Float64):
            out = out.with_columns(pl.col(name).cast(pl_type, strict=False).fill_null(0.0))
        elif pl_type in (pl.Int32, pl.Int64):
            out = out.with_columns(pl.col(name).cast(pl_type, strict=False).fill_null(0))
        elif pl_type == pl.Utf8:
            out = out.with_columns(pl.col(name).cast(pl_type, strict=False))

    out = out.select([f.name for f in schema])
    return out

def stage_collection_with_schema(
        out_path: str,
        iter_chunks_fn,
        canonical_schema: pa.Schema | None = None,
        resume: bool = True,
        compression: str = "zstd"
    ):
    __log.info(f"Starting staged write to {out_path}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ckpt = out_path + ".ckpt"

    target_schema = None

    if os.path.exists(out_path):
        try:
            target_schema = pq.ParquetFile(out_path).schema_arrow
            __log.info(f"Using existing file schema: {target_schema}")
        except Exception as e:
            __log.exception(f"Cannot read existing schema: {e}")
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
            table = pa.Table.from_arrays(
                [df[c].to_arrow() for c in target_schema.names], schema=target_schema
            )

        if writer is None:
            writer = pq.ParquetWriter(out_path, target_schema, compression= compression)
        writer.write_table(table)

        with open(ckpt, "w") as f:
            f.write(last_id)
        __log.info(f"Wrote chunk {i}, rows={len(df)}, checkpoint={last_id}")
        del table
        del df
        gc.collect()

    if writer:
         writer.close()

def stream_parquet(in_path: str, out_path: str, process_fn, *, chunk_rows: int = 200_000, compression: str = "zstd"):
    __log.info(f"Stream {in_path} -> {out_path} (rows/batch={chunk_rows})")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pf = pq.ParquetFile(in_path)
    writer = None

    for i, batch in enumerate(pf.iter_batches(batch_size=chunk_rows)):
        df = pl.from_arrow(batch)
        df = process_fn(df)
        tbl = df.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(out_path, tbl.schema, compression=compression)
        writer.write_table(tbl)
        del df, tbl, batch
        gc.collect()

    if writer:
        writer.close()
    __log.info(f"Completed streaming to {out_path}")

def sanitize_list_column(series: pl.Series, dtype: np.dtype) -> pl.Series:
    def _clean(val):
        try:
            arr = np.asarray(val, dtype=dtype)
            return arr.tolist()
        except Exception:
            return []

    pl_type = pl.List(pl.Float32 if dtype == np.float32 else pl.Float64)
    return series.map_elements(_clean).cast(pl_type)

def table_from_df_and_schema(df: pl.DataFrame, schema: pa.Schema) -> pa.Table:
    df = align_df_to_schema(df, schema)
    arrays = [df[name].to_arrow() for name in schema.names]
    return pa.Table.from_arrays(arrays=arrays, schema=schema)



