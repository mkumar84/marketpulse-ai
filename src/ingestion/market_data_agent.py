import pandas as pd
import requests
from src.utils.config import ALPHA_VANTAGE_API_KEY


def fetch_market_prices(symbols: list = None, days: int = 30) -> pd.DataFrame:
    """Fetch OHLCV data from Alpha Vantage. Returns empty DataFrame on any failure."""
    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOGL"]

    if not ALPHA_VANTAGE_API_KEY:
        return pd.DataFrame()

    base_url = "https://www.alphavantage.co/query"
    df_list = []

    for symbol in symbols:
        try:
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "apikey": ALPHA_VANTAGE_API_KEY,
            }

            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if "Error Message" in data or "Note" in data or "Information" in data:
                continue

            time_series = data.get("Time Series (Daily)", {})
            if not time_series:
                continue

            rows = []
            for date_str, values in list(time_series.items())[:days]:
                rows.append({
                    "date": pd.to_datetime(date_str),
                    "open": float(values.get("1. open", 0)),
                    "high": float(values.get("2. high", 0)),
                    "low": float(values.get("3. low", 0)),
                    "close": float(values.get("4. close", 0)),
                    "volume": float(values.get("5. volume", 0)),
                    "symbol": symbol,
                })

            if rows:
                symbol_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
                symbol_df["returns"] = symbol_df["close"].pct_change()
                # Adaptive window: use available rows (min 2) up to 20; annualise to yearly %
                vol_window = max(2, min(len(symbol_df), 20))
                symbol_df["volatility_pct"] = (
                    symbol_df["returns"].rolling(vol_window, min_periods=2).std() * (252 ** 0.5) * 100
                ).round(2)
                symbol_df["daily_range_pct"] = (
                    (symbol_df["high"] - symbol_df["low"]) / symbol_df["close"] * 100
                ).round(2)
                df_list.append(symbol_df)

        except Exception:
            continue

    if not df_list:
        return pd.DataFrame()

    result = pd.concat(df_list, ignore_index=True)
    return result.sort_values("date").reset_index(drop=True)
