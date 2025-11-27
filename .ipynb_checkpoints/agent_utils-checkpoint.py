import os
import pandas as pd
import numpy as np
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json
import calendar
import uuid
from google.adk.tools.tool_context import ToolContext

from typing import List, Dict, Any, Optional, Tuple, Any
import math
import datetime

import requests
from scipy.stats import skew, kurtosis, jarque_bera
import scipy.stats as stats


from dotenv import load_dotenv

# --- SHARED MEMORY ---
# This dictionary persists across all agents
DATA_CACHE: Dict[str, Any] = {}

def load_api_key(key_name):
    """
    Load API_KEY from environment variables or .env file.
    """

    # Load .env file
    load_dotenv()

    api_key = os.getenv(key_name)

    if not api_key:
        raise ValueError(
            f"{key_name} missing.\n"
            "Please set it in your environment variables or in a .env file."
        )

    os.environ[key_name] = api_key
    return api_key

# --- CORE TOOLS ---

def coins_info(coin_symbol: str, tool_context: ToolContext) -> dict:
    
    """
    Retrieve information about a cryptocurrency and evaluate whether it meets 
    predefined approval criteria.

    This function queries the CoinMarketCap API for the latest cryptocurrency
    listings, constructs a pandas DataFrame containing market information, and
    checks whether the requested coin exists. If the coin is found, it is
    evaluated according to three rules:

        1. Market cap must be greater than 1 billion USD.
        2. 24-hour trading volume must be greater than 1 million USD.
        3. The cryptocurrency must have existed for at least 5 years.

    Requires approval if Market cap < 1 billion USD or 24-hour trading volume < 1 million USD.

    Args:
        coin_symbol (str): The ticker symbol of the cryptocurrency (e.g., "BTC", "ETH").

    Returns:
        dict: A dictionary containing:
            - "symbol" (str): The requested symbol in uppercase.
            - "approved" (bool): Whether the coin satisfies all approval criteria.
            - "reasons" (list or str): List of reasons for rejection or approvation.

    """
    
    COINMARKETCAP_API_KEY =  load_api_key("COINMARKETCAP_API_KEY")
    
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
    parameters = {
      'start':'1',
      'limit':'5000'
    }
    headers = {
      'Accepts': 'application/json',
      'X-CMC_PRO_API_KEY': COINMARKETCAP_API_KEY,
    }
    
    session = Session()
    session.headers.update(headers)
    
    try:
      response = session.get(url, params=parameters)
      data = json.loads(response.text)

    except (ConnectionError, Timeout, TooManyRedirects) as e:
      print(e)
    
    df = pd.DataFrame([
    {
        'symbol': item['symbol'],
        'market_cap': item['quote']['USD']['market_cap'],
        'volume_24h': item['quote']['USD']['volume_24h'],
        'years_from_creation': (datetime.datetime.utcnow() - datetime.datetime.fromisoformat(item['date_added'].replace("Z", ""))).days / 365
    }
    for item in data['data']
    ])
    
    # CASE 1: The coin doesn't exist: the tool stops and return approved: False and reasons: The Coin is not present 
    if coin_symbol not in df["symbol"].values:
        return {
        "symbol": coin_symbol.upper(),
        "approved": False,
        "reasons": "The Coin is not present"}
        

    reasons = []

    coin = df[df.symbol == coin_symbol].iloc[0]

     # ---------- RULE 1: Market Cap > 1.000.000.000 ----------
    
    if coin.market_cap  < 1_000_000_000:
        reasons.append("Market cap < 1B")

    # ---------- RULE 2: Volume 24h > 1.000.000 ----------
        
    if coin.volume_24h < 1_000_000:
        reasons.append("24h volume < 1M")

    # ---------- RULE 3: Data added at least 5 years ago ----------

    if coin.years_from_creation < 5:
        reasons.append("Coin younger than 5 years")

    # ---------- FINAL RESULT ----------
    approved = len(reasons) == 0

    # CASE 2: The coin respect the 3 rules and it is approved
    if approved:

        return {
            "symbol": coin_symbol.upper(),
            "approved": approved,
            "reasons": "The Coin respect the 3 rules and it is approved"}

    # CASE 3: one or more of the rules is not meet so it is needed user approval. tool_context is called - PAUSE here.
    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(
            hint=f"⚠️ Reasons of request approval: {reasons}. Do you want to approve?",
            payload={"reasons": reasons},
        )
        return {  # This is sent to the Agent
            "symbol": coin_symbol.upper(),
            "approved": False,
            "reasons": f"Reasons for request approval: {reasons}",
        }

    # The tool_context is called AGAIN and is now resuming. Handle approval response - RESUME here.
    if tool_context.tool_confirmation.confirmed:
        return {
            "symbol": coin_symbol.upper(),
            "approved": True, 
            "reasons": "User Approved"}
    else:
        return {
            "symbol": coin_symbol.upper(),
            "approved": False,
            "reasons": "User Rejected"}


def get_intraday_data(
    ticker: str = "",
    exchange: str = "CC",
    interval: str = "1h",
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Retrieves full intraday historical data for a given ticker from EODHD in 120-day chunks.
    
    Parameters
    ----------
    ticker : str
        Crypto ticker symbol. The expected format is BASE-QUOTE, for example:
            - "BTC-USD"
            - "ETH-USD"
            - "BTC-EUR"

        If the user provides only the base ticker (e.g., "BTC", "ETH"),
        the tool automatically converts it to "<TICKER>-USD".
        
        Examples:
            "BTC"  → "BTC-USD"
            "ETH"  → "ETH-USD"
            
    exchange : str
        EODHD exchange code. Example: "CC" for crypto.
    interval : str
        Intraday interval (e.g., "1m", "5m", "1h").
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str, optional
        End date in YYYY-MM-DD format. Defaults to UTC now.
    
    Returns
    -------
    dict
        
        Success Example:
        {
              "status": "success",
              "data": {<list of dict rows>
              }
            }
        
        Error Example:
        {
            "status": "error",
            "error_message": "Invalid start_date format. Expected YYYY-MM-DD."
        }
    """

    # --- Load API Key ---
    try:
        eod_api_key = load_api_key("EODHD_API_KEY")
        if not eod_api_key:
            return {
                "status": "error",
                "error_message": "EODHD_API_KEY environment variable not set."
            }
    except Exception as e:
        return {"status": "error", "error_message": f"API key loading failed: {e}"}

    # --- Auto-correct ticker format ---
    # Expected: BASE-QUOTE (BTC-USD). If missing quote currency, default to USD.
    if "-" not in ticker:
        ticker = f"{ticker}-USD"

    # --- Date Validation ---
    try:
        start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError:
        return {
            "status": "error",
            "error_message": f"Invalid start_date '{start_date}'. Expected format YYYY-MM-DD."
        }

    try:
        end_dt = (
            datetime.datetime.utcnow()
            if end_date is None
            else datetime.datetime.strptime(end_date, "%Y-%m-%d")
        )
    except ValueError:
        return {
            "status": "error",
            "error_message": f"Invalid end_date '{end_date}'. Expected format YYYY-MM-DD."
        }

    if start_dt >= end_dt:
        return {
            "status": "error",
            "error_message": "start_date must be earlier than end_date."
        }

    all_chunks = []

    # --- Fetch Data in Rolling Windows ---
    try:
        while start_dt < end_dt:
            chunk_end_dt = min(start_dt + datetime.timedelta(days=120), end_dt)

            url = f"https://eodhistoricaldata.com/api/intraday/{ticker}.{exchange}"
            params = {
                "api_token": eod_api_key,
                "interval": interval,
                "fmt": "json",
                "from": calendar.timegm(start_dt.utctimetuple()),
                "to": calendar.timegm(chunk_end_dt.utctimetuple())
            }

            resp = requests.get(url, params=params)

            if resp.status_code != 200:
                return {
                    "status": "error",
                    "error_message": f"HTTP {resp.status_code} while fetching chunk {start_dt} → {chunk_end_dt}"
                }

            try:
                data_chunk = resp.json()
            except Exception:
                return {
                    "status": "error",
                    "error_message": "API returned non-JSON response."
                }

            # If chunk is not empty → convert to DataFrame
            if data_chunk:
                df_chunk = pd.DataFrame(data_chunk)

                if "datetime" not in df_chunk.columns:
                    return {
                        "status": "error",
                        "error_message": "API response missing required 'datetime' column."
                    }

                df_chunk["datetime"] = pd.to_datetime(df_chunk["datetime"])
                all_chunks.append(df_chunk)

            start_dt = chunk_end_dt

    except Exception as e:
        return {"status": "error", "error_message": f"Unexpected error: {e}"}

    # --- Combine All Data ---
    if not all_chunks:
        return {
            "status": "error",
            "error_message": "No data returned for the requested period."
        }

    df = pd.concat(all_chunks, ignore_index=True)
    df = df.drop_duplicates(subset="datetime")
    df = df.set_index("datetime").sort_index()

    # Remove fields not needed
    for col in ["timestamp", "gmtoffset"]:
        if col in df.columns:
            df = df.drop(columns=col)

    reference_id = str(uuid.uuid4())
    
    # Store the raw list of dicts in the global cache
    DATA_CACHE[reference_id] = df.reset_index().to_dict(orient="records")
    
    return {
        "status": "success",
        "reference_id": reference_id, # <--- The LLM will see this now
        "message": f"Data fetched successfully for {ticker}. Rows: {len(df)}"
    }


def get_ohlcv(
    ticker: str = "",
    exchange: str = "CC"
) -> dict:
    """
    Retrieve OHLCV historical daily data from EOD Historical Data API.

    Parameters
    ----------
    ticker : str
        Asset ticker symbol.  
        Crypto tickers MUST be in BASE-QUOTE format:
            - "BTC-USD"
            - "ETH-USD"

        If the user provides only the base ticker (e.g., "BTC"),
        the tool automatically converts it to "<TICKER>-USD".

    exchange : str
        Exchange code. For crypto use "CC". Default: "CC".


    Returns
    -------
    dict
            {
              "status": "success",
              "data": {<list of dict rows>
              }
            }
        OR
            {
              "status": "error",
              "error_message": "Explanation..."
            }
    """

    # ---------------------------------------
    # Validate API key
    # ---------------------------------------
    EODHD_API_KEY = load_api_key("EODHD_API_KEY")
    if not EODHD_API_KEY:
        return {
            "status": "error",
            "error_message": "EODHD_API_KEY environment variable not set."
        }

    # ---------------------------------------
    # Auto-correct ticker format (BTC → BTC-USD)
    # ---------------------------------------
    if "-" not in ticker:
        ticker = f"{ticker}-USD"

    # ---------------------------------------
    # Determine date range
    # ---------------------------------------
    period_end = datetime.datetime.today(),
    start = datetime.date(2000, 1, 1),

    # ---------------------------------------
    # API Request
    # ---------------------------------------
    url = f"https://eodhistoricaldata.com/api/eod/{ticker}.{exchange}"
    params = {
        "api_token": EODHD_API_KEY,
        "fmt": "json",
        "from": start,
        "to": period_end
    }

    try:
        resp = requests.get(url, params=params)
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Network request failed: {e}"
        }

    if resp.status_code != 200:
        return {
            "status": "error",
            "error_message": f"EODHD API returned HTTP {resp.status_code}"
        }

    # ---------------------------------------
    # Parse JSON
    # ---------------------------------------
    try:
        data_json = resp.json()
    except Exception:
        return {
            "status": "error",
            "error_message": "API returned invalid JSON."
        }

    if not data_json:
        return {
            "status": "error",
            "error_message": "No data returned from API."
        }

    # ---------------------------------------
    # Build DataFrame
    # ---------------------------------------
    try:
        df = pd.DataFrame(data_json)
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to convert JSON to DataFrame: {e}"
        }

    # Validate required columns
    if "date" not in df.columns:
        return {
            "status": "error",
            "error_message": "Missing 'date' field in API response."
        }

    # Clean DataFrame
    try:
        df.set_index("date", inplace=True)
        df.index.names = ['datetime']
        if "adjusted_close" in df.columns:
            df.drop("adjusted_close", axis=1, inplace=True)
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Data cleanup failed: {e}"
        }

    reference_id = str(uuid.uuid4())
    
    # Store the raw list of dicts in the global cache
    DATA_CACHE[reference_id] = df.reset_index().to_dict(orient="records")
    
    return {
        "status": "success",
        "reference_id": reference_id, # <--- The LLM will see this now
        "message": f"Data fetched successfully for {ticker}. Rows: {len(df)}"
    }


def clean_dataframe(
    df: dict,
    freq: str = "auto",
    repair_ohlc: bool = True
) -> dict:
    """
    Clean and validate a historical OHLCV DataFrame.

    The function inspects, repairs, and validates:

        ✔ Missing values
        ✔ Zero or negative OHLC values
        ✔ Zero or negative volume
        ✔ Missing timestamps (reindexing)
        ✔ Duplicate timestamps
        ✔ Out-of-order timestamps
        ✔ Infinite values
        ✔ Optional OHLC consistency repairs (e.g., low > high)

    Parameters
    ----------
    df : dict
    A dictionary containing a "data" field with a list of OHLCV rows.
    Example:
    {
        "data": [
            {"datetime": "2024-01-01", "open": 100, "high": 105, "low": 99, "close": 103, "volume": 12345},
            ...
        ]
    }

    
    freq : str
        Frequency of data ("1D", "1H", etc.).
        If "auto", the frequency is inferred from the median time difference.

    repair_ohlc : bool
        If True, ensures OHLC constraints:
            low ≤ open, close ≤ high

    Returns
    -------
    dict
       
        {
            "status": "success",
            "clean_df": <list of dict rows>,
            "report": [...]
        }
        OR
        {
            "status": "error",
            "error_message": "..."
        }
    """

    # ---------------------------
    # Validate input
    # ---------------------------
    dataframe = pd.DataFrame(df['data'])
    dataframe.set_index('datetime',inplace=True)
    if not isinstance(dataframe, pd.DataFrame):
        return {"status": "error", "error_message": "Input must be a pandas DataFrame."}

    if dataframe.empty:
        return {"status": "error", "error_message": "DataFrame is empty."}

    # Copy to avoid mutation
    data = dataframe.copy()
    report = []

    # ---------------------------
    # Ensure DatetimeIndex
    # ---------------------------
    try:
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
            report.append("Converted index to DatetimeIndex.")
    except Exception:
        return {"status": "error", "error_message": "Index cannot be converted to datetime."}

    # ---------------------------
    # Sort index
    # ---------------------------
    if not data.index.is_monotonic_increasing:
        data = data.sort_index()
        report.append("Sorted datetime index.")

    # ---------------------------
    # Remove duplicate timestamps
    # ---------------------------
    dupe_count = data.index.duplicated().sum()
    if dupe_count > 0:
        data = data[~data.index.duplicated(keep="first")]
        report.append(f"Removed {dupe_count} duplicate timestamps.")

    # ---------------------------
    # Infer frequency if needed
    # ---------------------------
    if freq == "auto":
        inferred = data.index.to_series().diff().median()
        if pd.isna(inferred):
            return {"status": "error", "error_message": "Cannot infer frequency."}
        freq = pd.tseries.frequencies.to_offset(inferred).freqstr
        report.append(f"Auto-detected data frequency as {freq}.")

    # ---------------------------
    # Detect missing timestamps
    # ---------------------------
    full_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq=freq)
    missing = full_index.difference(data.index)

    if len(missing) > 0:
        report.append(f"Found {len(missing)} missing timestamps.")
        data = data.reindex(full_index)
        report.append("Reindexed DataFrame to a complete timeline.")

    # ---------------------------
    # Handle infinite values
    # ---------------------------
    if np.isinf(data.values).any():
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        report.append("Replaced infinite values with NaN.")

    # ---------------------------
    # Detect & remove negative/zero OHLC
    # ---------------------------
    ohlc_cols = [c for c in data.columns if c.lower() in ["open", "high", "low", "close"]]
    for col in ohlc_cols:
        bad_count = (data[col] <= 0).sum()
        if bad_count > 0:
            data.loc[data[col] <= 0, col] = np.nan
            report.append(f"Column '{col}' had {bad_count} zero/negative values → set to NaN.")

    # ---------------------------
    # Fix bad OHLC relationships (optional)
    # ---------------------------
    if repair_ohlc:
        for idx, row in data[ohlc_cols].iterrows():
            if row.isna().any():
                continue

            low, high = row["low"], row["high"]
            open_, close = row["open"], row["close"]

            broken = False
            if low > high:
                low, high = min(low, high), max(low, high)
                broken = True
            if open_ > high:
                open_ = high
                broken = True
            if close > high:
                close = high
                broken = True
            if open_ < low:
                open_ = low
                broken = True
            if close < low:
                close = low
                broken = True

            if broken:
                data.at[idx, "low"] = low
                data.at[idx, "high"] = high
                data.at[idx, "open"] = open_
                data.at[idx, "close"] = close
                report.append(f"Repaired inconsistent OHLC at {idx}.")

    # ---------------------------
    # Clean volume column
    # ---------------------------
    if "volume" in data.columns:
        neg_vol = (data["volume"] < 0).sum()

        if neg_vol > 0:
            data.loc[data["volume"] < 0, "volume"] = np.nan
            report.append(f"Volume had {neg_vol} negative values → replaced with NaN.")

    # ---------------------------
    # Interpolate missing values
    # ---------------------------
    missing_after = data.isna().sum().sum()
    if missing_after > 0:
        data.interpolate(method="time", inplace=True)
        data.fillna(method="ffill", inplace=True)
        data.fillna(method="bfill", inplace=True)
        report.append("Interpolated remaining missing values (time, ffill, bfill).")

    # ---------------------------
    # SUCCESS: return clean data
    # ---------------------------
    return {
        "status": "success",
        "clean_df": data.reset_index().to_dict(orient="records"),
        "report": report
    }


def calculate_max_drawdown(prices):
    """Calculate maximum drawdown"""
    cumulative = prices / prices.cummax()
    return (1 - cumulative.min())

def return_distribution_analysis(df: dict, time_1: int = 60, time_2: int = 252) -> dict:
    """
    Analyzes return distribution and returns a dictionary of statistics 
    for the LLM to interpret.
    """
    # Data Prep
    data = pd.DataFrame(df['data'])
    data.set_index('datetime', inplace=True)
    data.index = pd.to_datetime(data.index)
    
    # Calculate Returns
    dataframe = pd.DataFrame(data['close'].copy())
    dataframe['returns'] = np.log(dataframe / dataframe.shift(1))
    dataframe.dropna(inplace=True)

    # --- Helper to build stats dict for a specific window ---
    def get_stats(window_df, prices):
        ext_threshold = 2 * window_df.std()
        return {
            "mean_daily_return_pct": round(window_df.mean() * 100, 4),
            "annualized_volatility_pct": round(window_df.std() * np.sqrt(365) * 100, 4),
            "skewness": round(skew(window_df), 4),
            "kurtosis": round(kurtosis(window_df), 4),
            "extreme_up_moves_pct": round(len(window_df[window_df > ext_threshold]) / len(window_df) * 100, 2),
            "extreme_down_moves_pct": round(len(window_df[window_df < -ext_threshold]) / len(window_df) * 100, 2),
            "max_drawdown_pct": round(calculate_max_drawdown(prices) * 100, 2),
            "normality_p_value": round(stats.jarque_bera(window_df)[1], 4)
        }

    # Calculate stats for both timeframes
    stats_short = get_stats(dataframe['returns'].tail(time_1), dataframe['close'].tail(time_1))
    stats_long = get_stats(dataframe['returns'].tail(time_2), dataframe['close'].tail(time_2))

    # Note: We keep the plotting logic if you run this locally to see the graph, 
    # but strictly speaking, the Agent doesn't see the graph, only the dict below.
    
    return {
        "status": "success",
        "short_term_stats": stats_short,
        "long_term_stats": stats_long,
        "analysis_note": "High Kurtosis (>3) implies fat tails (mean reversion potential). Positive Skew implies trend following potential."
    }





