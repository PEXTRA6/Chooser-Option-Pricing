from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf
from pandas_datareader import data as web


START_DATE = "2018-01-01"
END_DATE = date.today().isoformat()
OUTPUT_FILE = Path("week1_initial_raw_dataset.csv")


def clean_yfinance_columns(
    data: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """
    Flatten yfinance columns and retain the requested ticker.
    This supports both ordinary columns and MultiIndex columns.
    """
    if data.empty:
        raise ValueError(f"No data was downloaded for {ticker}.")

    if isinstance(data.columns, pd.MultiIndex):
        # yfinance may return columns such as ('Close', 'JPM').
        if ticker in data.columns.get_level_values(-1):
            data = data.xs(ticker, axis=1, level=-1)
        else:
            data.columns = [
                "_".join(
                    str(value)
                    for value in column
                    if str(value)
                )
                for column in data.columns
            ]

    data = data.copy()
    data.columns = [
        str(column).strip().replace(" ", "_")
        for column in data.columns
    ]

    return data


def download_market_data() -> pd.DataFrame:
    print(f"Downloading market data from {START_DATE} to {END_DATE}.")

    # JPM stock market data.
    jpm = yf.download(
        "JPM",
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        progress=False,
    )

    jpm = clean_yfinance_columns(jpm, "JPM")

    required_jpm_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj_Close",
        "Volume",
    ]

    available_jpm_columns = [
        column
        for column in required_jpm_columns
        if column in jpm.columns
    ]

    if "Close" not in available_jpm_columns:
        raise ValueError(
            "JPM Close data was not found in the Yahoo Finance result."
        )

    jpm = jpm[available_jpm_columns]

    # VIX market index.
    vix = yf.download(
        "^VIX",
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        progress=False,
    )

    vix = clean_yfinance_columns(vix, "^VIX")

    if "Close" not in vix.columns:
        raise ValueError(
            "VIX Close data was not found in the Yahoo Finance result."
        )

    vix = vix[["Close"]].rename(columns={"Close": "VIX"})

    # FRED 10-year Treasury constant maturity rate.
    treasury = web.DataReader(
        "DGS10",
        "fred",
        START_DATE,
        END_DATE,
    )

    treasury = treasury.rename(
        columns={"DGS10": "Treasury_10Y"}
    )

    # Combine all datasets using the date index.
    raw_data = jpm.join(vix, how="outer")
    raw_data = raw_data.join(treasury, how="outer")

    raw_data.index = pd.to_datetime(raw_data.index)
    raw_data.index.name = "Date"

    raw_data = raw_data.sort_index()

    # Treasury data may contain non-trading-day gaps.
    raw_data["Treasury_10Y"] = (
        raw_data["Treasury_10Y"].ffill()
    )

    # Keep rows where JPM traded.
    raw_data = raw_data.dropna(subset=["Close"])

    raw_data = raw_data.reset_index()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("Week 1 data download completed.")
    print(f"Rows generated: {len(raw_data)}")
    print(f"Latest date: {raw_data['Date'].max()}")
    print(f"Saved to: {OUTPUT_FILE.resolve()}")

    return raw_data


if __name__ == "__main__":
    dataset = download_market_data()
    print(dataset.tail())
