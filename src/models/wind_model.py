import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "data/processed/wind/wind_featured.csv"

df = pd.read_csv(DATA_PATH)

df['Date'] = pd.to_datetime(df['Date'])
df = pd.get_dummies(df, columns=['Season'], prefix='Season')

# Sort properly
df = df.sort_values(by=['State', 'Date']).reset_index(drop=True)
print("Data loaded and sorted temporally by State.")

# Lag features
df['PS_lag_1'] = df.groupby('State')['PS'].shift(1)
df['T2M_lag_1'] = df.groupby('State')['T2M'].shift(1)
df['RH2M_lag_1'] = df.groupby('State')['RH2M'].shift(1)
df['Wind_Power_lag_1'] = df.groupby('State')['Wind_Power_Density'].shift(1)

# Target
target_variable = 'Target_Next_Day_Wind_Power'
df[target_variable] = df.groupby('State')['Wind_Power_Density'].shift(-1)

df = df.dropna()

season_cols = [col for col in df.columns if col.startswith("Season_")]

features = [
    'Latitude', 'Longitude', 'PS', 'PS_lag_1', 'T2M', 'T2M_lag_1',
    'RH2M', 'RH2M_lag_1', 'PRECTOTCORR', 'ALLSKY_SFC_SW_DWN',
    'WS50M', 'Wind_Power_lag_1'
] + season_cols

X = df[features]
y = df[target_variable]

print(f"Target (y): {target_variable}")
print(f"Final Feature Set (X): {len(features)} features")

# Time-based split
cutoff_date = '2023-01-01'

train_mask = df['Date'] < cutoff_date
test_mask = df['Date'] >= cutoff_date

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
    'Random Forest': RandomForestRegressor(random_state=42, n_estimators=50, max_depth=10),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42, n_estimators=50),
    'LightGBM': LGBMRegressor(random_state=42, n_estimators=50, max_depth=10, verbosity=-1),
    'Extra Trees': ExtraTreesRegressor(random_state=42, n_estimators=50, max_depth=10),
    'Bagging Regressor': BaggingRegressor(random_state=42, n_estimators=10),
    'KNN': KNeighborsRegressor(n_neighbors=5)
}

results = []

print("Starting Model Comparison...\n")

for name, model in models.items():
    print(f"Training {name}...")
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results.append({
        'Model': name,
        'R2 Score': r2,
        'RMSE': rmse,
        'MAE': mae
    })

results_df = pd.DataFrame(results).sort_values(by='R2 Score', ascending=False)

print("\n=== Model Comparison Complete ===")
print(results_df)

plt.figure(figsize=(12, 6))
sns.barplot(x='R2 Score', y='Model', data=results_df)
plt.title('Model Comparison: R² Score for Next-Day Wind Power Forecasting')
plt.xlabel('R² Score')
plt.ylabel('Algorithm')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

best_model_name = results_df.iloc[0]['Model']
print(f"\nThe overall best performing model is: {best_model_name}")