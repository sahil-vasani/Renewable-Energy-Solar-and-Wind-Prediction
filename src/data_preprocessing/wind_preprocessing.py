import pandas as pd
import os

INPUT_FILE = "data/raw/wind/wind_raw.csv"
OUTPUT_FILE = "data/processed/wind/wind_clean.csv"

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


def load_data():
    print("Loading wind data...")
    return pd.read_csv(INPUT_FILE)


def handle_missing_values(df):
    print("Handling missing values...")
    df = df.fillna(method="ffill").fillna(method="bfill")
    return df


def remove_duplicates(df):
    print("Removing duplicates...")
    return df.drop_duplicates()


def feature_engineering(df):
    print("Creating new features...")

    df["Date"] = pd.to_datetime(df["Date"])

    # Time features
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day

    # Wind-specific feature
    if "WS50M" in df.columns:
        df["wind_power_estimate"] = df["WS50M"] ** 3  # wind power ~ v^3

    return df


def save_data(df):
    print("Saving cleaned wind data...")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved at: {OUTPUT_FILE}")


def preprocess_wind():
    df = load_data()
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = feature_engineering(df)
    save_data(df)

 