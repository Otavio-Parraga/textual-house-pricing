from custom_metrics import median_absolute_percentage_error
from lightgbm import LGBMRegressor
from math import sqrt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, KFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import pandas as pd


df = pd.read_json(f'../datasets/preprocessed/homes.json')

print('Dataset has {:d} samples.'.format(len(df)))

print(df.groupby('type')['id'].count().sort_values(ascending=False))

df = df[df['type'] != 'lots land']

print(f'Dataframe has now {len(df)} samples')

df = df.dropna()

print(f'Dataframe has now {len(df)} samples')

print(df.dtypes)

# Keep only relevant features
df = df[[
    'area',
    'rooms',
    'bathrooms',
    'age',
    # 'lat',
    # 'lng',
    'price',
]]

# pd.set_option('float_format', '{:f}'.format)

# print(df.describe().T)

# We don't have missing data

# missing_ratio = df.isna().mean().sort_values(ascending=False)

# print('Features with missing data:')
# print(missing_ratio)
# print(missing_ratio[missing_ratio > 0])

regressors = {
    'Extremely Randomized Trees': ExtraTreesRegressor(n_jobs=-1, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_jobs=-1, random_state=42),
    'XGBoost': XGBRegressor(n_jobs=-1, random_state=42),
    'LightGBM': LGBMRegressor(n_jobs=-1, random_state=42),
}

for name, regressor in regressors.items():
    print('Evaluating regressor {:s}...'.format(name))

    pipeline = Pipeline(steps=[
        ('regressor', regressor),
    ])

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    X, y = df.drop(columns='price'), df['price']

    results = cross_validate(pipeline, X=X, y=y, cv=kfold, scoring={
        'mae': make_scorer(mean_absolute_error),
        'mape': make_scorer(mean_absolute_percentage_error),
        'mdape': make_scorer(median_absolute_percentage_error),
        'mse': make_scorer(mean_squared_error),
        'r2': make_scorer(r2_score),
    })

    print('MSE: {:.3f} | MAE: {:.3f} | RMSE: {:.3f} | MAPE: {:.3f} | MdAPE: {:.3f} | R2: {:.3f}'.format(
        results['test_mse'].mean(),
        results['test_mae'].mean(),
        sqrt(results['test_mse'].mean()),
        results['test_mape'].mean(),
        results['test_mdape'].mean(),
        results['test_r2'].mean(),
    ))
