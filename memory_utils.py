import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce overall memory footprint.

    The function converts float64 and int64 columns to float32 and int32
    respectively.  It modifies ``df`` in place and also returns it for
    convenience so it can be used inline.  A small log message is emitted
    describing the before/after sizes.
    """
    start_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    float_cols = df.select_dtypes(include=["float64"]).columns
    int_cols = df.select_dtypes(include=["int64"]).columns
    if len(float_cols):
        df[float_cols] = df[float_cols].astype(np.float32)
    if len(int_cols):
        df[int_cols] = df[int_cols].astype(np.int32)
    end_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logger.info(f"[mem] dataframe downcast {start_mb:.2f}MB -> {end_mb:.2f}MB")
    return df
