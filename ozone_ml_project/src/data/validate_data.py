import pandas as pd

def validate_dataframe(df: pd.DataFrame) -> None:
    """
    Perform basic checks to ensure the dataset is model-ready.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index must be a DatetimeIndex.")

    if df.index.isna().sum() > 0:
        raise ValueError("DatetimeIndex contains NaT values.")

    if not df.index.is_monotonic_increasing:
        raise ValueError("DatetimeIndex is not sorted in increasing order.")

    if df.index.duplicated().sum() > 0:
        raise ValueError("Duplicate timestamps found in index.")

    if df.isna().sum().sum() > 0:
        raise ValueError("Missing values still present in dataframe.")