import pandas as pd 
import numpy as np 

df = pd.read_csv("data/processed/solar/solar_featured.csv")

print(df.head())

df = df.drop(columns=['Date','State'])

X = df.drop(columns=['ALLSKY_SFC_SW_DWN'])
y = df['ALLSKY_SFC_SW_DWN']

numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

base_models = [
    ('ridge', Ridge(alpha=1.0)),
    ('rf', RandomForestRegressor(n_estimators=200, random_state=42)),
    ('xgb', XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42))
]

stack_model = StackingRegressor(
    estimators=base_models,
    final_estimator=Ridge()    
)

from sklearn.ensemble import BaggingRegressor, VotingRegressor

bag = BaggingRegressor(
    n_estimators=50,
    random_state=42,
    n_jobs=1
)

rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=1
)

xgb = XGBRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=1,
    verbosity=0
)

voting_model = VotingRegressor(
    estimators=[
        ('bag', bag),
        ('rf',  rf),
        ('xgb', xgb)
    ],
    n_jobs=1
)

model_pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', voting_model)
])

model_2_pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', stack_model)
])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_2_pipeline.fit(X_train, y_train)

print("Stacking Model R²:", model_2_pipeline.score(X_test, y_test))

from sklearn.metrics import r2_score, mean_absolute_error

model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

print("Voting Ensemble R²:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

param_distributions = {
    'model__rf__n_estimators': [80,110],          
    'model__rf__max_depth': [10,16,15,None],
    
    'model__xgb__n_estimators': [100, 200, 300],
    'model__xgb__max_depth': [5,7,10],
    'model__xgb__learning_rate': [0.1, 0.05,0.03],      
    'model__xgb__subsample': [0.8,1.0],
    
    'model__bag__n_estimators': [30,40]
}

from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    estimator=model_pipeline,
    param_distributions=param_distributions,
    n_iter=20,          
    cv=3,
    scoring='r2',
    verbose=2,
    random_state=42,
    n_jobs=1             
)

random_search.fit(X_train, y_train)

print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)