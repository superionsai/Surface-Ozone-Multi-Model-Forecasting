import pandas as pd

def load_processed_data(path: str) -> pd.DataFrame:
    """
    Load processed dataset and handle both:
    - Timestamp as column
    - Timestamp already saved as index
    """

    df = pd.read_csv(path)

    # Case 1: Timestamp is a column
    if "Timestamp" in df.columns:

        df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="mixed", errors="coerce")
        df = df.dropna(subset=["Timestamp"])
        df["Timestamp"] = df["Timestamp"].dt.normalize()
        df = df.set_index("Timestamp")

    # Case 2: Timestamp already index (saved CSV)
    else:
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.dropna()

    # Remove unwanted columns
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Sort index
    df = df.sort_index()

    return df