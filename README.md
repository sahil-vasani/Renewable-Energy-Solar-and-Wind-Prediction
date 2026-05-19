# 🌞🌬️ Renewable Energy Prediction — Solar & Wind

> A production-grade Machine Learning pipeline for forecasting solar irradiance and wind power density across all **36 States and Union Territories of India**, powered by decade-long NASA meteorological data.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Motivation](#-motivation)
- [Data Source](#️-data-source)
- [Repository Structure](#-repository-structure)
- [Pipeline Architecture](#-pipeline-architecture)
- [Feature Engineering](#-feature-engineering)
- [ML Models](#-ml-models)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Dataset Variables](#-dataset-variables)
- [Key Results & Highlights](#-key-results--highlights)
- [Technologies Used](#-technologies-used)
- [Future Work](#-future-work)
- [License](#-license)

---

## 🔭 Project Overview

India's renewable energy sector is among the fastest-growing in the world, yet predicting energy yield remains a critical challenge due to complex meteorological dependencies. This project builds **twin end-to-end ML pipelines** — one for **Solar** energy and one for **Wind** energy — to forecast renewable energy potential using 10 years of historical daily weather data (2015–2024).

Each pipeline covers the complete data science lifecycle: **data ingestion → preprocessing → feature engineering → EDA → model training & evaluation**.

---

## 💡 Motivation

- Renewable energy output is highly sensitive to atmospheric conditions (solar angle, cloud cover, wind velocity, air density, etc.).
- Accurate forecasting enables better **grid management**, **investment decisions**, and **energy policy planning**.
- India's diverse geography — from Rajasthan's arid solar belt to Tamil Nadu's coastal wind corridors — makes state-level modelling essential.

---

## 🛰️ Data Source

**NASA POWER API** — [Prediction Of Worldwide Energy Resources](https://power.larc.nasa.gov/)

- **Temporal Coverage**: January 1, 2015 → December 31, 2024 (daily resolution)
- **Spatial Coverage**: 36 Indian States and Union Territories
- **Access Method**: REST API (`/api/temporal/daily/point`), Community: `RE` (Renewable Energy)

### Solar Parameters Fetched

| Parameter | Description |
|---|---|
| `ALLSKY_SFC_SW_DWN` | All-Sky Surface Shortwave Downward Irradiance (primary target proxy) |
| `TOA_SW_DWN` | Top-of-Atmosphere Shortwave Downward Irradiance |
| `CLOUD_AMT` | Cloud Amount (fraction) |
| `AOD_55` | Aerosol Optical Depth at 550 nm |
| `SZA` | Solar Zenith Angle |
| `PW` | Precipitable Water |
| `QV2M` | Specific Humidity at 2m |
| `T2M` | Temperature at 2m (°C) |
| `RH2M` | Relative Humidity at 2m (%) |
| `WS50M` | Wind Speed at 50m (m/s) |
| `PS` | Surface Pressure (kPa) |

### Wind Parameters Fetched

| Parameter | Description |
|---|---|
| `WS10M` | Wind Speed at 10m (m/s) |
| `WS50M` | Wind Speed at 50m (m/s) — primary feature |
| `WD10M` | Wind Direction at 10m |
| `T2M` | Temperature at 2m (°C) |
| `T2M_MAX` / `T2M_MIN` | Daily Max/Min Temperature at 2m |
| `T2MDEW` | Dew Point Temperature at 2m |
| `RH2M` | Relative Humidity at 2m (%) |
| `PS` | Surface Pressure (kPa) |
| `PRECTOTCORR` | Bias-Corrected Precipitation (mm/day) |
| `ALLSKY_SFC_SW_DWN` | Solar Irradiance (cross-pipeline feature) |

---

## 📁 Repository Structure

```
Renewable-Energy-Solar-and-Wind-Prediction/
│
├── data/
│   ├── raw/
│   │   ├── solar/
│   │   │   └── solar_raw.csv                  # Raw merged solar dataset (~12 MB)
│   │   └── wind/
│   │       └── wind_raw.csv                   # Raw merged wind dataset (~13 MB)
│   └── processed/
│       ├── solar/
│       │   ├── solar_clean.csv                # After preprocessing (~16 MB)
│       │   └── solar_featured.csv             # After feature engineering (~20 MB)
│       └── wind/
│           └── wind_clean.csv                 # After preprocessing (~20 MB)
│
├── src/
│   ├── data_ingestion/
│   │   ├── solar_data_loader.py               # Fetches & merges solar data via NASA API
│   │   └── wind_data_loader.py                # Fetches & merges wind data via NASA API
│   │
│   ├── data_preprocessing/
│   │   ├── solar_preprocessing.py             # Cleans, imputes & creates base features
│   │   └── wind_preprocessing.py              # Cleans, imputes & creates wind base features
│   │
│   ├── feature_engineering/
│   │   ├── solar_features.py                  # Lag, rolling, time-based features for solar
│   │   └── wind_features.py                   # Lag, rolling, physics-based features for wind
│   │
│   └── models/
│       ├── solar_model.py                     # Voting Ensemble + Stacking + Hyperparameter Tuning
│       └── wind_model.py                      # Multi-model benchmark with time-based split
│
├── EDA/
│   ├── 01_eda_solar.ipynb                     # Deep-dive solar EDA (~2.6 MB with outputs)
│   └── 02_eda_wind.ipynb                      # Spatial, temporal & correlation EDA for wind
│
├── main_pipeline.py                           # Orchestrates full solar + wind pipeline
├── .gitignore
└── README.md
```

---

## 🏗️ Pipeline Architecture

The project runs as two independent but structurally parallel pipelines, both orchestrated by `main_pipeline.py`.

```
NASA POWER API
      │
      ▼
[Data Ingestion]          solar_data_loader.py / wind_data_loader.py
  ↳ Fetch per state (36)
  ↳ Merge into master CSV
      │
      ▼
[Preprocessing]           solar_preprocessing.py / wind_preprocessing.py
  ↳ Forward/backward fill missing values
  ↳ Deduplicate records
  ↳ Parse and extract date components (year, month, day)
  ↳ Compute base energy estimates
      │
      ▼
[Feature Engineering]     solar_features.py / wind_features.py
  ↳ Temporal features (day_of_week, is_weekend)
  ↳ Lag features (t-1, t-7)
  ↳ Rolling window mean (7-day)
  ↳ Physics-based features (wind_power_estimate = WS50M³)
      │
      ▼
[Model Training]          solar_model.py / wind_model.py
  ↳ Train/test split (random or time-based)
  ↳ Preprocessing pipeline (impute + scale + encode)
  ↳ Model training and evaluation
  ↳ Hyperparameter tuning (RandomizedSearchCV)
```

---

## 🧪 Feature Engineering

### Solar Features

| Feature | Description |
|---|---|
| `year`, `month`, `day` | Extracted from `Date` |
| `day_of_week` | Numerical day of week (0=Mon) |
| `is_weekend` | Binary flag: 1 if Saturday/Sunday |
| `solar_lag_1` | `ALLSKY_SFC_SW_DWN` shifted by 1 day |
| `solar_lag_7` | `ALLSKY_SFC_SW_DWN` shifted by 7 days |
| `solar_roll_mean_7` | 7-day rolling average of irradiance |
| `solar_energy_estimate` | `ALLSKY_SFC_SW_DWN × 0.2` (proxy energy yield) |

### Wind Features

| Feature | Description |
|---|---|
| `year`, `month`, `day` | Extracted from `Date` |
| `day_of_week` | Numerical day of week |
| `is_weekend` | Binary weekend flag |
| `wind_lag_1` | `WS50M` shifted by 1 day |
| `wind_lag_7` | `WS50M` shifted by 7 days |
| `wind_roll_mean_7` | 7-day rolling mean of wind speed at 50m |
| `wind_power_estimate` | `WS50M³` — proportional to kinetic energy in wind |
| `PS_lag_1`, `T2M_lag_1`, `RH2M_lag_1` | State-grouped 1-day lags for pressure, temp, humidity |
| `Wind_Power_lag_1` | 1-day lag of Wind Power Density (grouped by state) |
| `Target_Next_Day_Wind_Power` | `shift(-1)` of Wind Power Density — **prediction target** |

> ⚠️ **Leakage Prevention**: All lag operations in the wind model are computed with `df.groupby('State')` to ensure no cross-state temporal contamination.

---

## 🤖 ML Models

### Solar Model (`solar_model.py`)

The solar pipeline implements two ensemble strategies and optimizes via randomized search.

**Voting Ensemble (Primary)**
- `BaggingRegressor` (n=50)
- `RandomForestRegressor` (n=200)
- `XGBRegressor` (n=200)

**Stacking Ensemble (Secondary)**
- Base: `Ridge`, `RandomForestRegressor`, `XGBRegressor`
- Meta-learner: `Ridge`

**Preprocessing Pipeline (sklearn)**
- Numeric: `SimpleImputer(mean)` → `StandardScaler`
- Categorical: `SimpleImputer(most_frequent)` → `OneHotEncoder`

**Hyperparameter Tuning**
- `RandomizedSearchCV` with 3-fold CV, 20 iterations
- Search space: RF depth/estimators, XGB learning rate/subsample/depth, Bagging estimators

---

### Wind Model (`wind_model.py`)

The wind pipeline runs a **comprehensive multi-model benchmark** with a strict time-based train/test split.

**Train/Test Split**
- Training: all data before `2023-01-01`
- Testing: data from `2023-01-01` onward
- Purpose: simulates real-world next-day forecasting without temporal leakage

**Models Evaluated**

| Model | Notes |
|---|---|
| Linear Regression | Baseline |
| Ridge Regression | Regularised linear baseline |
| Decision Tree | `max_depth=10` |
| Random Forest | `n_estimators=50`, `max_depth=10` |
| Gradient Boosting | `n_estimators=50` |
| **LightGBM** | `n_estimators=50`, `max_depth=10` |
| Extra Trees | `n_estimators=50`, `max_depth=10` |
| Bagging Regressor | `n_estimators=10` |
| KNN | `n_neighbors=5` |

**Evaluation Metrics**: R² Score, RMSE, MAE — results ranked in a comparative leaderboard.

---

## 📊 Exploratory Data Analysis

### Solar EDA (`01_eda_solar.ipynb`)
- Distribution of solar irradiance by state and season
- Correlation analysis between `ALLSKY_SFC_SW_DWN`, cloud cover, solar zenith angle, humidity
- Identification of highest solar potential states (e.g., Rajasthan, Gujarat)
- Time-series plots of irradiance trends (2015–2024)

### Wind EDA (`02_eda_wind.ipynb`)
- **Spatial Analysis**: State-wise bar chart of average Wind Power Density
- **Temporal/Seasonal Analysis**: Monthly boxplots revealing monsoon-season peaks (June–September)
- **Autocorrelation**: ACF/PACF plots for a high-potential state (Gujarat) to identify lag dependencies
- **Correlation Heatmap**: Feature correlations with Wind Power Density

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet access (for NASA API data ingestion)

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/Renewable-Energy-Solar-and-Wind-Prediction.git
cd Renewable-Energy-Solar-and-Wind-Prediction
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

If a `requirements.txt` is not yet present, install manually:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn statsmodels requests
```

---

## 🚀 Usage

### Option A: Run the Full Pipeline (Recommended)

```bash
python main_pipeline.py
```

This sequentially executes:
1. Solar preprocessing → feature engineering → model training
2. Wind preprocessing → feature engineering → model training

### Option B: Run Each Stage Independently

```bash
# 1. Fetch raw data from NASA API (run once)
python src/data_ingestion/solar_data_loader.py
python src/data_ingestion/wind_data_loader.py

# 2. Preprocess
python src/data_preprocessing/solar_preprocessing.py
python src/data_preprocessing/wind_preprocessing.py

# 3. Feature Engineering
python src/feature_engineering/solar_features.py
python src/feature_engineering/wind_features.py

# 4. Train Models
python src/models/solar_model.py
python src/models/wind_model.py
```

### Option C: Explore via Jupyter Notebooks

```bash
jupyter notebook EDA/01_eda_solar.ipynb
jupyter notebook EDA/02_eda_wind.ipynb
```

---

## 📐 Dataset Variables

### Geographic Coverage

All 36 Indian States and Union Territories are covered, including:

**States (28)**: Andhra Pradesh, Arunachal Pradesh, Assam, Bihar, Chhattisgarh, Goa, Gujarat, Haryana, Himachal Pradesh, Jharkhand, Karnataka, Kerala, Madhya Pradesh, Maharashtra, Manipur, Meghalaya, Mizoram, Nagaland, Odisha, Punjab, Rajasthan, Sikkim, Tamil Nadu, Telangana, Tripura, Uttar Pradesh, Uttarakhand, West Bengal

**Union Territories (8)**: Andaman and Nicobar Islands, Chandigarh, Dadra and Nagar Haveli and Daman and Diu, Delhi, Jammu and Kashmir, Ladakh, Lakshadweep, Puducherry

### Data Schema (Processed CSVs)

```
Date         | datetime  | Daily date stamp
State        | string    | Indian state/UT name
Latitude     | float     | Decimal latitude of state capital
Longitude    | float     | Decimal longitude of state capital
<NASA vars>  | float     | Meteorological parameters (see table above)
year         | int       | Extracted year
month        | int       | Extracted month (1–12)
day          | int       | Extracted day of month
*_lag_1      | float     | 1-day lag of target variable
*_lag_7      | float     | 7-day lag of target variable
*_roll_mean_7| float     | 7-day rolling mean
```

---

## 🏆 Key Results & Highlights

- **Leakage-Free Forecasting**: The wind model uses `shift(-1)` for target construction and a strict temporal train/test split (pre/post 2023-01-01), making it a genuine next-day forecasting system.
- **Geospatial Integrity**: Lag features are computed within state boundaries using `groupby('State')`, preventing geographical data leakage.
- **Comprehensive Benchmarking**: 9 regression algorithms are evaluated side-by-side on the wind task, with an automated R²-ranked leaderboard.
- **Dual Ensemble Strategy**: The solar model compares both hard Voting and Stacking ensembles under identical preprocessing pipelines to identify the optimal architecture.
- **Decade-Scale Dataset**: ~3,650 daily records per state × 36 states = ~131,400 rows per domain, providing robust statistical power.

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| Data Manipulation | pandas, numpy |
| Machine Learning | scikit-learn, XGBoost, LightGBM |
| Visualisation | matplotlib, seaborn |
| Time Series Analysis | statsmodels (ACF/PACF) |
| Data Ingestion | requests (NASA POWER REST API) |
| Notebooks | Jupyter |
| Version Control | Git |

---

## 🔮 Future Work

- [ ] Add a `requirements.txt` and optionally a `Dockerfile` for reproducibility
- [ ] Implement LSTM/GRU deep learning models for temporal sequence modelling
- [ ] Extend feature engineering with Wind Power Density (`½ × ρ × v³`) using computed air density from temperature and pressure
- [ ] Add a geospatial dashboard (Folium / Plotly Dash) for interactive state-level energy potential mapping
- [ ] Introduce cross-validation with `TimeSeriesSplit` for the solar model to honour temporal ordering
- [ ] Package the pipeline as a CLI tool with configurable `--domain`, `--state`, and `--date-range` flags
- [ ] Publish model performance metrics (R², RMSE, MAE) in this README after final runs

---

## 🙏 Acknowledgements

- [NASA POWER Project](https://power.larc.nasa.gov/) for providing free, high-quality meteorological data via open API
- The scikit-learn, XGBoost, and LightGBM open-source communities

---

*Built with ☀️ and 🌬️ — harnessing data to power India's renewable energy future.*