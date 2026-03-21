# Renewable Energy Predictive Modeling: Solar & Wind

A comprehensive Machine Learning project built to analyze and forecast **Solar** and **Wind** Energy potential across all the 36 states and Union Territories in India using historical and geographical meteorological data.

---

## ⚡ Project Overview

Renewable energy output is highly dependent on meteorological conditions and geolocation. This project bridges data science and renewable energy by fetching daily historical weather data from the **NASA POWER API** (2015-2024).

The project is structured into twin pipelines—one strictly for Solar analysis and one for Wind analysis. Each pipeline encompasses data extraction, feature engineering, exploratory data analysis (EDA), and rigorous machine learning forecasting.

---

## 📁 Repository Structure

### ☀️ Solar Energy Pipeline
- **`Fetch and Preprocessing(Solar)/`**
  - `1_data_fetch_solar.ipynb`: Fetches historical insolation, temperature, and atmospheric data from the NASA API.
  - `2_solar_preprocessing.ipynb`: Cleans data and generates `Solar_Preprocessed.csv`.
- **`EDA/`**
  - `EDA(Solar).ipynb`: Explores correlations between solar irradiance, humidity, clear sky conditions, and identifies the best states for solar farms.
- **`Model(Solar)/`**
  - `All_model.ipynb`: Trains and compares regression algorithms for solar potential prediction.
  - `Selected_model.ipynb`: Fine-tunes the ultimate choice model for Solar.

### 🌬️ Wind Energy Pipeline
- **`Fetch and Preprocessing (Wind)/`**
  - `Data Fetch.ipynb`: Fetches wind velocities (10M, 50M), temperature, and pressure data.
  - `preprocess.ipynb`: Engineers critical domain features like `Air_Density` and `Wind_Power_Density` to create `India_Renewable_Energy_MASTER_DATASET_Calculated.csv`.
- **`EDA/`**
  - `EDA(Wind).ipynb`: Analyzes seasonality (e.g., higher wind power during monsoon months), Autocorrelation (ACF/PACF) for lagged dependencies, and spatial mappings across Indian states.
- **`Model(Wind)/`**
  - `All_model.ipynb`: A robust, leakage-free pipeline that uses `shift(-1)` to forecast **Next-Day Wind Power Density**. Features an automated evaluator ranking models like LightGBM, Random Forest, and Gradient Boosting by $R^2$ and RMSE scores.

---

## 🛰️ Data Source

- **NASA Prediction Of Worldwide Energy Resources (POWER) API**
- Both pipelines fetch continuous time-series data for parameters like:
  - **Solar Variables**: `ALLSKY_SFC_SW_DWN` (Solar Irradiance), `CLRSKY_SFC_SW_DWN` (Clear Sky Irradiance).
  - **Wind Variables**: `WS10M`, `WS50M` (Wind Velocities), `WD10M` (Wind Direction).
  - **General Weather**: `T2M` (Temperature), `RH2M` (Humidity), `PS` (Surface Pressure), `PRECTOTCORR` (Precipitation).

---

## 💡 Key Machine Learning Highlights
- **Leakage-Free Temporal Modeling**: The forecasting models are designed using shift operators to accurately map *today's* weather conditions to *tomorrow's* renewable energy yield, creating a genuine predictive tool rather than a retrospective formula.
- **Geospatial Integrity**: Feature engineering appropriately groups structural lags (like previous day's pressure) strictly within State boundaries to prevent geographical data corruption.
- **Comprehensive Benchmarking**: `All_model.ipynb` scripts loop through almost every continuous machine learning algorithm available in `scikit-learn` and `lightgbm`, ensuring the chosen architecture is mathematically optimal.

---

## 🚀 Setup & Execution

### Prerequisites
- Python 3.8+
- Essential Libraries: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `matplotlib`, `seaborn`, `statsmodels`.

### Execution Order
1. **Data Acquisition**: Run the fetch and preprocessing notebooks inside the respective Solar/Wind fetch directories.
2. **Analysis**: Run the notebooks situated in the `EDA/` directory to glean insights.
3. **Forecasting**: Run the models within `Model(Solar)/` or `Model(Wind)/` to observe the performance comparison rankings!
