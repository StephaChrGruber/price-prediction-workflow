import logging

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

logger = logging.getLogger(__name__)


def _downcast_float(series: pd.Series) -> pd.Series:
    """Downcast a float ``Series`` as much as possible."""

    f16 = series.astype(np.float16)
    if np.allclose(series, f16, equal_nan=True):
        return f16
    return series.astype(np.float32)


def _downcast_int(series: pd.Series) -> pd.Series:
    """Downcast an integer ``Series`` to the smallest possible subtype."""

    c_min, c_max = series.min(), series.max()
    if c_min >= 0:
        # unsigned
        if c_max <= np.iinfo(np.uint8).max:
            return series.astype(np.uint8)
        if c_max <= np.iinfo(np.uint16).max:
            return series.astype(np.uint16)
        if c_max <= np.iinfo(np.uint32).max:
            return series.astype(np.uint32)
        return series.astype(np.uint64)
    # signed
    if np.iinfo(np.int8).min <= c_min and c_max <= np.iinfo(np.int8).max:
        return series.astype(np.int8)
    if np.iinfo(np.int16).min <= c_min and c_max <= np.iinfo(np.int16).max:
        return series.astype(np.int16)
    if np.iinfo(np.int32).min <= c_min and c_max <= np.iinfo(np.int32).max:
        return series.astype(np.int32)
    return series.astype(np.int64)


def reduce_mem_usage(df: pd.DataFrame, *, cat_threshold: float = 0.5) -> pd.DataFrame:
    """Downcast numeric columns and optionally convert objects to categories.

    Parameters
    ----------
    df:
        Dataframe to be modified in-place.
    cat_threshold:
        Convert an ``object`` column to ``category`` if its number of unique
        values divided by the total number of rows is below this threshold.

    Returns
    -------
    ``pd.DataFrame``
        The modified dataframe.  The same object is also modified in-place.

    Notes
    -----
    Logs the memory usage before and after the optimisation and returns the
    dataframe so it can be used inline.
    """

    start_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

    for col in df.columns:
        col_data = df[col]
        col_type = col_data.dtype

        if ptypes.is_bool_dtype(col_type):
            df[col] = col_data.astype(bool)
        elif ptypes.is_integer_dtype(col_type):
            if col_data.dropna().isin([0, 1]).all():
                df[col] = col_data.astype(bool)
            else:
                df[col] = _downcast_int(col_data)
        elif ptypes.is_float_dtype(col_type):
            col_no_na = col_data.dropna()
            if not col_data.isna().any() and (col_no_na == col_no_na.astype(np.int64)).all():
                df[col] = _downcast_int(col_data.astype(np.int64))
            else:
                df[col] = _downcast_float(col_data)
        elif ptypes.is_object_dtype(col_type):
            num_unique = col_data.nunique(dropna=False)
            if col_data.dropna().isin([0, 1, "0", "1", True, False, "True", "False"]).all():
                df[col] = col_data.map({"0": False, "1": True, 0: False, 1: True, "True": True, "False": False}).astype(bool)
            elif num_unique / len(col_data) <= cat_threshold:
                df[col] = col_data.astype("category")

    end_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logger.info(f"[mem] dataframe downcast {start_mb:.2f}MB -> {end_mb:.2f}MB")
    return df
