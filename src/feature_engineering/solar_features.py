import pandas as pd
import os

INPUT_FILE = "data/processed/solar/solar_clean.csv"
OUTPUT_FILE = "data/processed/solar/solar_featured.csv"

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


def load_data():
    print("Loading solar cleaned data...")
    return pd.read_csv(INPUT_FILE)


def create_time_features(df):
    print("Creating time-based features...")

    df["Date"] = pd.to_datetime(df["Date"])

    df["day_of_week"] = df["Date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df


def create_lag_features(df):
    print("Creating lag features...")

    if "ALLSKY_SFC_SW_DWN" in df.columns:
        df["solar_lag_1"] = df["ALLSKY_SFC_SW_DWN"].shift(1)
        df["solar_lag_7"] = df["ALLSKY_SFC_SW_DWN"].shift(7)

    return df


def create_rolling_features(df):
    print("Creating rolling features...")

    if "ALLSKY_SFC_SW_DWN" in df.columns:
        df["solar_roll_mean_7"] = df["ALLSKY_SFC_SW_DWN"].rolling(window=7).mean()

    return df


def handle_missing_after_features(df):
    print("Handling missing values after feature engineering...")
    return df.fillna(method="bfill").fillna(method="ffill")


def save_data(df):
    print("Saving solar featured data...")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved at: {OUTPUT_FILE}")


def feature_engineering_solar():
    df = load_data()
    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = handle_missing_after_features(df)
    save_data(df)

 