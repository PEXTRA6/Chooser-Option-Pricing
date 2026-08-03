from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")


def load_jpm_dividend_history(end_date: pd.Timestamp) -> pd.DataFrame:
    ticker = yf.Ticker("JPM")
    dividends = ticker.dividends

    if dividends.empty:
        raise ValueError("No JPM dividend history was returned.")

    dividends = dividends.copy()
    dividends.index = pd.to_datetime(dividends.index)

    if dividends.index.tz is not None:
        dividends.index = dividends.index.tz_localize(None)

    dividend_df = (
        dividends.loc[dividends.index <= end_date]
        .rename("Dividend")
        .reset_index()
    )

    dividend_df.columns = ["Date", "Dividend"]
    dividend_df["Date"] = (
        pd.to_datetime(dividend_df["Date"])
        .dt.tz_localize(None)
        .dt.normalize()
    )

    return dividend_df


def build_week2_pipeline(input_file: str, output_file: str) -> pd.DataFrame:
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path)

    if "Adj_Close" in df.columns:
        df = df.rename(columns={"Adj_Close": "Adj Close"})

    df["Date"] = (
        pd.to_datetime(df["Date"], errors="coerce")
        .dt.tz_localize(None)
        .dt.normalize()
    )

    df = (
        df.dropna(subset=["Date"])
          .sort_values("Date")
          .drop_duplicates(subset="Date")
          .reset_index(drop=True)
    )

    numeric_cols = [
        "Open","High","Low","Close","Adj Close",
        "Volume","VIX","Treasury_10Y"
    ]

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    # Traditional features
    df["Daily_Return"] = df["Adj Close"].pct_change()
    df["Log_Return"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))
    df["VIX_Change"] = df["VIX"].pct_change()
    df["Rate_Change"] = df["Treasury_10Y"].diff()
    df["Volume_Change"] = df["Volume"].pct_change()

    for w in [5,20,60]:
        df[f"Rolling_Vol_{w}D"] = (
            df["Log_Return"].rolling(w).std()*np.sqrt(252)
        )

    # Dividend
    dividend_df = load_jpm_dividend_history(df["Date"].max())

    df = pd.merge_asof(
        df.sort_values("Date"),
        dividend_df.sort_values("Date"),
        on="Date",
        direction="backward"
    )

    df["Dividend"] = df["Dividend"].ffill().fillna(0)

    df["Dividend_Growth"] = 0.0
    changed = df["Dividend"].ne(df["Dividend"].shift(1))
    prev = df["Dividend"].shift(1)
    mask = changed & prev.gt(0)
    df.loc[mask, "Dividend_Growth"] = (
        df.loc[mask, "Dividend"] / prev.loc[mask] - 1
    )

    # Advanced features
    df["VIX_Return"] = df["VIX"].pct_change()

    df["VIX_JPM_Correlation_20D"] = (
        df["Daily_Return"]
        .rolling(20)
        .corr(df["VIX_Return"])
    )

    df["Rate_Momentum_1D"] = df["Treasury_10Y"].diff(1)
    df["Rate_Momentum_5D"] = df["Treasury_10Y"].diff(5)
    df["Rate_Momentum_20D"] = df["Treasury_10Y"].diff(20)

    df["Rate_Pct_Change_5D"] = df["Treasury_10Y"].pct_change(5)
    df["Rate_Pct_Change_20D"] = df["Treasury_10Y"].pct_change(20)

    vmin = df["VIX"].min()
    vmax = df["VIX"].max()
    if np.isclose(vmin, vmax):
        df["Market_Sentiment_Index"] = 0.5
    else:
        df["Market_Sentiment_Index"] = (
            1 - (df["VIX"]-vmin)/(vmax-vmin)
        ).clip(0,1)

    df = df.replace([np.inf,-np.inf], np.nan)

    keep = [
        "Date","Open","High","Low","Close","Adj Close","Volume",
        "VIX","Treasury_10Y","Daily_Return","Log_Return",
        "VIX_Change","Rate_Change","Volume_Change",
        "Rolling_Vol_5D","Rolling_Vol_20D","Rolling_Vol_60D",
        "Dividend","Dividend_Growth","Market_Sentiment_Index",
        "VIX_Return","VIX_JPM_Correlation_20D",
        "Rate_Momentum_1D","Rate_Momentum_5D","Rate_Momentum_20D",
        "Rate_Pct_Change_5D","Rate_Pct_Change_20D"
    ]

    df = (
        df.dropna(subset=[
            "Rolling_Vol_60D",
            "VIX_JPM_Correlation_20D",
            "Rate_Momentum_20D",
            "Rate_Pct_Change_20D"
        ])
        .loc[:, keep]
        .reset_index(drop=True)
    )

    quality_report = pd.DataFrame({
        "Column": df.columns,
        "Data_Type": df.dtypes.astype(str),
        "Missing_Values": df.isna().sum().values,
        "Missing_Percentage": (df.isna().mean()*100).round(4).values,
        "Unique_Values": df.nunique().values
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    quality_report.to_csv(
        output_path.parent/"week2_quality_report.csv",
        index=False
    )

    print("Week2 dataset saved:", output_path)
    return df


if __name__ == "__main__":
    INPUT_FILE = "data/week1_initial_raw_dataset.csv"
    OUTPUT_FILE = "output/week2_feature_dataset_pipeline.csv"
    week2 = build_week2_pipeline(INPUT_FILE, OUTPUT_FILE)
    print(week2.head())
