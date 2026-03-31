import requests
import pandas as pd
import io
import os
import glob

OUTPUT_DIR = "data/raw/wind"
FINAL_FILE = "data/raw/wind/wind_raw.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FINAL_FILE), exist_ok=True)


def fetch_wind_data():
    states = [
        {"name": "Andhra Pradesh", "lat": 16.5062, "lon": 80.6480},
    {"name": "Arunachal Pradesh", "lat": 27.1004, "lon": 93.6056},
    {"name": "Assam", "lat": 26.1445, "lon": 91.7362},
    {"name": "Bihar", "lat": 25.5941, "lon": 85.1376},
    {"name": "Chhattisgarh", "lat": 21.2514, "lon": 81.6296},
    {"name": "Goa", "lat": 15.4909, "lon": 73.8278},
    {"name": "Gujarat", "lat": 23.2156, "lon": 72.6369},
    {"name": "Haryana", "lat": 30.7333, "lon": 76.7794},
    {"name": "Himachal Pradesh", "lat": 31.1048, "lon": 77.1734},
    {"name": "Jharkhand", "lat": 23.3441, "lon": 85.3096},
    {"name": "Karnataka", "lat": 12.9716, "lon": 77.5946},
    {"name": "Kerala", "lat": 8.5241, "lon": 76.9366},
    {"name": "Madhya Pradesh", "lat": 23.2599, "lon": 77.4126},
    {"name": "Maharashtra", "lat": 19.0760, "lon": 72.8777},
    {"name": "Manipur", "lat": 24.8170, "lon": 93.9368},
    {"name": "Meghalaya", "lat": 25.5788, "lon": 91.8933},
    {"name": "Mizoram", "lat": 23.7367, "lon": 92.7176},
    {"name": "Nagaland", "lat": 25.6751, "lon": 94.1086},
    {"name": "Odisha", "lat": 20.2961, "lon": 85.8245},
    {"name": "Punjab", "lat": 30.7333, "lon": 76.7794},
    {"name": "Rajasthan", "lat": 26.9124, "lon": 75.7873},
    {"name": "Sikkim", "lat": 27.3389, "lon": 88.6065},
    {"name": "Tamil Nadu", "lat": 13.0827, "lon": 80.2707},
    {"name": "Telangana", "lat": 17.3850, "lon": 78.4867},
    {"name": "Tripura", "lat": 23.8315, "lon": 91.2868},
    {"name": "Uttar Pradesh", "lat": 26.8467, "lon": 80.9462},
    {"name": "Uttarakhand", "lat": 30.3165, "lon": 78.0322},
    {"name": "West Bengal", "lat": 22.5726, "lon": 88.3639},
 
    {"name": "Andaman and Nicobar Islands", "lat": 11.6234, "lon": 92.7265},
    {"name": "Chandigarh", "lat": 30.7333, "lon": 76.7794},
    {"name": "Dadra and Nagar Haveli and Daman and Diu", "lat": 20.4283, "lon": 72.8397},
    {"name": "Delhi", "lat": 28.6139, "lon": 77.2090},
    {"name": "Jammu and Kashmir", "lat": 34.0837, "lon": 74.7973},
    {"name": "Ladakh", "lat": 34.1526, "lon": 77.5771},
    {"name": "Lakshadweep", "lat": 10.5667, "lon": 72.6417},
    {"name": "Puducherry", "lat": 11.9416, "lon": 79.8083}
    ]

    parameters = [
        "WS10M", "WS50M", "PS", "T2M", "WD10M",
        "T2M_MAX", "T2M_MIN", "T2MDEW", "RH2M",
        "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"
    ]

    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    start_date = "20150101"
    end_date = "20241231"

    for state in states:
        print(f"Fetching Wind Data for {state['name']}...")

        params = {
            "parameters": ",".join(parameters),
            "community": "RE",
            "longitude": state["lon"],
            "latitude": state["lat"],
            "start": start_date,
            "end": end_date,
            "format": "CSV"
        }

        try:
            response = requests.get(base_url, params=params, timeout=60)

            if response.status_code == 200:
                content = response.text
                idx = content.find("YEAR,MO,DY")

                if idx != -1:
                    df = pd.read_csv(io.StringIO(content[idx:]))

                    df["Date"] = pd.to_datetime(
                        df[["YEAR", "MO", "DY"]].astype(str).agg("-".join, axis=1)
                    )

                    df["State"] = state["name"]
                    df["Latitude"] = state["lat"]
                    df["Longitude"] = state["lon"]

                    df.drop(columns=["YEAR", "MO", "DY"], inplace=True)

                    filename = os.path.join(OUTPUT_DIR, f"{state['name']}_wind.csv")
                    df.to_csv(filename, index=False)

                    print(f"Saved: {filename}")

            else:
                print(f"Error {response.status_code} for {state['name']}")

        except Exception as e:
            print(f"Error for {state['name']}: {e}")


def merge_wind_data():
    files = glob.glob(os.path.join(OUTPUT_DIR, "*.csv"))

    if not files:
        print("No wind files found!")
        return

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["State", "Date"])

    df.to_csv(FINAL_FILE, index=False)

    print(f"Wind master dataset saved at: {FINAL_FILE}")


if __name__ == "__main__":
    fetch_wind_data()
    merge_wind_data()