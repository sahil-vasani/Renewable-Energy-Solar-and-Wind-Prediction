import pandas as pd
import os

INPUT_FILE = "data/processed/wind/wind_clean.csv"
OUTPUT_FILE = "data/processed/wind/wind_featured.csv"

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


def load_data():
    print("Loading wind cleaned data...")
    return pd.read_csv(INPUT_FILE)


def create_time_features(df):
    print("Creating time-based features...")

    df["Date"] = pd.to_datetime(df["Date"])

    df["day_of_week"] = df["Date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df


def create_lag_features(df):
    print("Creating lag features...")

    if "WS50M" in df.columns:
        df["wind_lag_1"] = df["WS50M"].shift(1)
        df["wind_lag_7"] = df["WS50M"].shift(7)

    return df


def create_rolling_features(df):
    print("Creating rolling features...")

    if "WS50M" in df.columns:
        df["wind_roll_mean_7"] = df["WS50M"].rolling(window=7).mean()

    return df


def create_physics_features(df):
    print("Creating physics-based features...")

    if "WS50M" in df.columns:
        df["wind_power_estimate"] = df["WS50M"] ** 3  # v^3 relation

    return df


def handle_missing_after_features(df):
    print("Handling missing values after feature engineering...")
    return df.fillna(method="bfill").fillna(method="ffill")


def save_data(df):
    print("Saving wind featured data...")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved at: {OUTPUT_FILE}")


def feature_engineering_wind():
    df = load_data()
    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_physics_features(df)
    df = handle_missing_after_features(df)
    save_data(df)


 