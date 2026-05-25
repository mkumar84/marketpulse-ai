import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "sample_market_data.csv"

def load_sample_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)
