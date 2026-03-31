import pandas as pd
import os

INPUT_FILE = "data/raw/solar/solar_raw.csv"
OUTPUT_FILE = "data/processed/solar/solar_clean.csv"

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)


def load_data():
    print("Loading solar data...")
    return pd.read_csv(INPUT_FILE)


def handle_missing_values(df):
    print("Handling missing values...")
    df = df.ffill().bfill()
    return df


def remove_duplicates(df):
    print("Removing duplicates...")
    return df.drop_duplicates()


def feature_engineering(df):
    print("Creating new features...")

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

    # Time features
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day

    # Example feature (can adjust based on your dataset)
    if "ALLSKY_SFC_SW_DWN" in df.columns:
        df["solar_energy_estimate"] = df["ALLSKY_SFC_SW_DWN"] * 0.2

    return df


def save_data(df):
    print("Saving cleaned solar data...")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved at: {OUTPUT_FILE}")


def preprocess_solar():
    df = load_data()
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = feature_engineering(df)
    save_data(df)


if __name__ == "__main__":
    preprocess_solar()