import pandas as pd
from src.config import Config

def load_japan_activities(cfg: Config) -> pd.DataFrame:
    """
    Load the Japan_Activites.csv dataset.
    Assumes: Activity column is text, all other columns are metadata.
    """
    if not cfg.RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {cfg.RAW_DATA_PATH}")

    df = pd.read_csv(cfg.RAW_DATA_PATH)

    if "Activity" not in df.columns:
        raise ValueError(f"Expected column 'Activity'. Got: {list(df.columns)}")

    # normalize column names (optional)
    df.columns = [c.strip() for c in df.columns]

    # drop totally empty rows
    df = df.dropna(how="all").reset_index(drop=True)

    return df
