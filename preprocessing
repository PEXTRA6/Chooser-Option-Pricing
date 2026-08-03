from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")


def load_jpm_dividend_history(
    end_date: pd.Timestamp,
) -> pd.DataFrame:
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
        .astype("datetime64[ns]")
    )

    return dividend_df


def build_week2_pipeline(
    input_file: str,
    output_file: str,
    figures_dir: str | None = None,
    generate_visualizations: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Automated Week 2 preprocessing and feature-engineering pipeline.

    The pipeline performs:
    1. Date and numeric-type cleaning
    2. Missing-value handling and time ordering
    3. Traditional and advanced feature engineering
    4. JPM dividend and dividend-growth integration
    5. VIX-derived Market Sentiment Index construction
    6. Correlation analysis
    7. Random-Forest feature-importance analysis
    8. Dataset, analysis tables, and figure export

    Notes
    -----
    Market_Sentiment_Index is a VIX-derived market-risk sentiment proxy,
    not a news-text sentiment score. A higher VIX produces a lower index.
    """

    input_path = Path(input_file)
    output_path = Path(output_file)

    if figures_dir is None:
        figures_path = output_path.parent / "figures"
    else:
        figures_path = Path(figures_dir)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file was not found: {input_path.resolve()}"
        )

    # --------------------------------------------------
    # 1. Load and standardize the raw dataset
    # --------------------------------------------------
    df = pd.read_csv(input_path)

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    if "Adj_Close" in df.columns and "Adj Close" not in df.columns:
        df = df.rename(columns={"Adj_Close": "Adj Close"})

    required_raw_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "VIX",
        "Treasury_10Y",
    ]

    missing_required = [
        column for column in required_raw_columns
        if column not in df.columns
    ]

    if missing_required:
        raise KeyError(
            "The following required columns are missing: "
            + ", ".join(missing_required)
        )

    df["Date"] = (
        pd.to_datetime(df["Date"], errors="coerce")
        .dt.tz_localize(None)
        .dt.normalize()
        .astype("datetime64[ns]")
    )

    df = (
        df.dropna(subset=["Date"])
        .sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "VIX",
        "Treasury_10Y",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Time-aligned market variables are carried forward only when missing.
    df[numeric_columns] = df[numeric_columns].ffill().bfill()

    # --------------------------------------------------
    # 2. Traditional financial features
    # --------------------------------------------------
    df["Daily_Return"] = df["Adj Close"].pct_change()
    df["Log_Return"] = np.log(
        df["Adj Close"] / df["Adj Close"].shift(1)
    )

    df["VIX_Change"] = df["VIX"].pct_change()
    df["Rate_Change"] = df["Treasury_10Y"].diff()
    df["Volume_Change"] = df["Volume"].pct_change()

    for window in [5, 20, 60]:
        df[f"Rolling_Vol_{window}D"] = (
            df["Log_Return"]
            .rolling(window=window)
            .std()
            * np.sqrt(252)
        )

    # --------------------------------------------------
    # 3. Dividend and dividend-growth features
    # --------------------------------------------------
    dividend_df = load_jpm_dividend_history(
        end_date=df["Date"].max()
    )

    df["Date"] = df["Date"].astype("datetime64[ns]")
    dividend_df["Date"] = dividend_df["Date"].astype(
        "datetime64[ns]"
    )

    df = pd.merge_asof(
        df.sort_values("Date"),
        dividend_df.sort_values("Date"),
        on="Date",
        direction="backward",
    )

    df["Dividend"] = df["Dividend"].ffill().fillna(0.0)

    df["Dividend_Growth"] = 0.0
    dividend_changed = df["Dividend"].ne(df["Dividend"].shift(1))
    previous_dividend = df["Dividend"].shift(1)

    valid_growth = dividend_changed & previous_dividend.gt(0)

    df.loc[valid_growth, "Dividend_Growth"] = (
        df.loc[valid_growth, "Dividend"]
        / previous_dividend.loc[valid_growth]
        - 1
    )

    # --------------------------------------------------
    # 4. Advanced features
    # --------------------------------------------------
    df["VIX_Return"] = df["VIX"].pct_change()

    df["VIX_JPM_Correlation_20D"] = (
        df["Daily_Return"]
        .rolling(window=20)
        .corr(df["VIX_Return"])
    )

    df["Rate_Momentum_1D"] = df["Treasury_10Y"].diff(1)
    df["Rate_Momentum_5D"] = df["Treasury_10Y"].diff(5)
    df["Rate_Momentum_20D"] = df["Treasury_10Y"].diff(20)

    df["Rate_Pct_Change_5D"] = df["Treasury_10Y"].pct_change(5)
    df["Rate_Pct_Change_20D"] = df["Treasury_10Y"].pct_change(20)

    # VIX-derived Market Sentiment Index in [0, 1].
    # Higher VIX -> lower market sentiment.
    vix_min = df["VIX"].min()
    vix_max = df["VIX"].max()

    if pd.isna(vix_min) or pd.isna(vix_max):
        raise ValueError("VIX contains no valid observations.")

    if np.isclose(vix_max, vix_min):
        df["Market_Sentiment_Index"] = 0.5
    else:
        df["Market_Sentiment_Index"] = 1.0 - (
            (df["VIX"] - vix_min)
            / (vix_max - vix_min)
        )

    df["Market_Sentiment_Index"] = (
        df["Market_Sentiment_Index"].clip(0.0, 1.0)
    )

    df = df.replace([np.inf, -np.inf], np.nan)

    exact_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "VIX",
        "Treasury_10Y",
        "Daily_Return",
        "Log_Return",
        "VIX_Change",
        "Rate_Change",
        "Volume_Change",
        "Rolling_Vol_5D",
        "Rolling_Vol_20D",
        "Rolling_Vol_60D",
        "Dividend",
        "Dividend_Growth",
        "Market_Sentiment_Index",
        "VIX_Return",
        "VIX_JPM_Correlation_20D",
        "Rate_Momentum_1D",
        "Rate_Momentum_5D",
        "Rate_Momentum_20D",
        "Rate_Pct_Change_5D",
        "Rate_Pct_Change_20D",
    ]

    df = (
        df.dropna(
            subset=[
                "Rolling_Vol_60D",
                "VIX_JPM_Correlation_20D",
                "Rate_Momentum_20D",
                "Rate_Pct_Change_20D",
            ]
        )
        .loc[:, exact_columns]
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # 5. Data-quality report
    # --------------------------------------------------
    quality_report = pd.DataFrame({
        "Column": df.columns,
        "Data_Type": df.dtypes.astype(str).values,
        "Missing_Values": df.isna().sum().values,
        "Missing_Percentage": (
            df.isna().mean().values * 100
        ).round(4),
        "Unique_Values": df.nunique().values,
    })

    # --------------------------------------------------
    # 6. Correlation analysis
    # --------------------------------------------------
    analysis_features = [
        "Daily_Return",
        "Log_Return",
        "VIX_Change",
        "Rate_Change",
        "Volume_Change",
        "Rolling_Vol_5D",
        "Rolling_Vol_20D",
        "Rolling_Vol_60D",
        "Dividend",
        "Dividend_Growth",
        "Market_Sentiment_Index",
        "VIX_Return",
        "VIX_JPM_Correlation_20D",
        "Rate_Momentum_1D",
        "Rate_Momentum_5D",
        "Rate_Momentum_20D",
        "Rate_Pct_Change_5D",
        "Rate_Pct_Change_20D",
    ]

    correlation_matrix = df[analysis_features].corr()

    # --------------------------------------------------
    # 7. Feature-importance analysis
    # --------------------------------------------------
    # The target is next-day absolute return, a simple future volatility proxy.
    # shift(-1) ensures that features at time t predict an outcome at t+1.
    importance_data = df.copy()
    importance_data["Next_Day_Absolute_Return"] = (
        importance_data["Daily_Return"].shift(-1).abs()
    )

    # Raw VIX is intentionally omitted because Market_Sentiment_Index is a
    # deterministic inverse transformation of VIX. Including both would duplicate
    # the same information and split Random-Forest importance between them.
    importance_features = [
        "VIX_Change",
        "Rate_Change",
        "Volume_Change",
        "Rolling_Vol_5D",
        "Rolling_Vol_20D",
        "Rolling_Vol_60D",
        "Dividend",
        "Dividend_Growth",
        "Market_Sentiment_Index",
        "VIX_Return",
        "VIX_JPM_Correlation_20D",
        "Rate_Momentum_1D",
        "Rate_Momentum_5D",
        "Rate_Momentum_20D",
        "Rate_Pct_Change_5D",
        "Rate_Pct_Change_20D",
    ]

    model_data = importance_data[
        importance_features + ["Next_Day_Absolute_Return"]
    ].dropna()

    if len(model_data) < 100:
        raise ValueError(
            "Not enough complete observations for feature-importance analysis."
        )

    X = model_data[importance_features]
    y = model_data["Next_Day_Absolute_Return"]

    importance_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )
    importance_model.fit(X, y)

    feature_importance = (
        pd.DataFrame({
            "Feature": importance_features,
            "Importance": importance_model.feature_importances_,
        })
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    # Correlation with the future target is kept separate from the full matrix.
    target_correlations = (
        model_data[importance_features]
        .corrwith(model_data["Next_Day_Absolute_Return"])
        .sort_values(key=lambda values: values.abs(), ascending=False)
        .rename("Correlation_with_Next_Day_Absolute_Return")
        .reset_index()
        .rename(columns={"index": "Feature"})
    )

    # --------------------------------------------------
    # 8. Save dataset and analysis tables
    # --------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    quality_report.to_csv(
        output_path.parent / "week2_quality_report.csv",
        index=False,
    )
    correlation_matrix.to_csv(
        output_path.parent / "week2_correlation_matrix.csv"
    )
    target_correlations.to_csv(
        output_path.parent / "week2_target_correlations.csv",
        index=False,
    )
    feature_importance.to_csv(
        output_path.parent / "week2_feature_importance.csv",
        index=False,
    )

    # --------------------------------------------------
    # 9. Visualization export
    # --------------------------------------------------
    if generate_visualizations:
        # 9.1 Correlation heatmap
        fig, ax = plt.subplots(figsize=(14, 11))
        image = ax.imshow(
            correlation_matrix,
            aspect="auto",
            vmin=-1,
            vmax=1,
        )
        fig.colorbar(image, ax=ax, label="Pearson correlation")
        ax.set_xticks(range(len(correlation_matrix.columns)))
        ax.set_xticklabels(
            correlation_matrix.columns,
            rotation=60,
            ha="right",
        )
        ax.set_yticks(range(len(correlation_matrix.index)))
        ax.set_yticklabels(correlation_matrix.index)
        ax.set_title("Week 2 Feature Correlation Matrix")
        fig.tight_layout()
        fig.savefig(
            figures_path / "week2_correlation_heatmap.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.show()
        plt.close(fig)

        # 9.2 Correlations with the future target
        plot_target_corr = target_correlations.sort_values(
            "Correlation_with_Next_Day_Absolute_Return"
        )
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(
            plot_target_corr["Feature"],
            plot_target_corr[
                "Correlation_with_Next_Day_Absolute_Return"
            ],
        )
        ax.axvline(0.0, linewidth=1)
        ax.set_xlabel("Pearson correlation")
        ax.set_title(
            "Feature Correlation with Next-Day Absolute Return"
        )
        fig.tight_layout()
        fig.savefig(
            figures_path / "week2_target_correlation.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.show()
        plt.close(fig)

        # 9.3 Random-Forest feature importance
        plot_importance = feature_importance.sort_values("Importance")
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(
            plot_importance["Feature"],
            plot_importance["Importance"],
        )
        ax.set_xlabel("Random-Forest importance")
        ax.set_title(
            "Feature Importance for Next-Day Absolute Return"
        )
        fig.tight_layout()
        fig.savefig(
            figures_path / "week2_feature_importance.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.show()
        plt.close(fig)

        # 9.4 VIX-derived sentiment over time
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df["Date"], df["Market_Sentiment_Index"])
        ax.set_xlabel("Date")
        ax.set_ylabel("Index (0–1)")
        ax.set_title("VIX-Derived Market Sentiment Index")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(
            figures_path / "week2_market_sentiment_index.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.show()
        plt.close(fig)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Dataset saved to: {output_path.resolve()}")
    print(f"Analysis tables saved to: {output_path.parent.resolve()}")
    print(f"Figures saved to: {figures_path.resolve()}")

    display(quality_report)
    display(target_correlations)
    display(feature_importance)

    return df


INPUT_FILE = "data/week1_initial_raw_dataset.csv"
OUTPUT_FILE = "output/week2_feature_dataset_pipeline.csv"
FIGURES_DIR = "output/figures"

week2_data_pipeline = build_week2_pipeline(
    input_file=INPUT_FILE,
    output_file=OUTPUT_FILE,
    figures_dir=FIGURES_DIR,
    generate_visualizations=True,
    random_state=42,
)

display(week2_data_pipeline.head())
