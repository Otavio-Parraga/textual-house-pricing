import pandas as pd
import numpy as np
import argparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from utils import median_absolute_percentage_error, relevant_features
from sklearn.model_selection import KFold, cross_validate
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score


def create_text_pipe(args):
    counter = CountVectorizer(
        binary=args.binary, min_df=args.min_df, dtype=np.float32)

    if args.n_components > 0 and args.tfidf == False:
        return make_pipeline(counter, TruncatedSVD(n_components=args.n_components, random_state=42))
    elif args.n_components > 0 and args.tfidf == True:
        return make_pipeline(counter, TfidfTransformer(), TruncatedSVD(n_components=args.n_components, random_state=42))
    elif args.n_components == 0 and args.tfidf == True:
        return make_pipeline(counter, TfidfTransformer())
    else:
        return make_pipeline(counter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str,
                        choices=['rent', 'sale'], required=True, help='available datasets are rent and sale')
    parser.add_argument('-t', '--text_only', action='store_true', help='if set, the model will use only textual information')
    parser.add_argument('-b', '--binary', action='store_true', help='make count vectorization binary')
    parser.add_argument('-n', '--n_components', type=int, default=0, help='n_components > 0 will apply dimensionality reduction with TruncatedSVD using n_components')
    parser.add_argument('-tf', '--tfidf', action='store_true', help='apply tfidf after vectorization')
    parser.add_argument('-md', '--min_df', type=float, default=0.001, help='minimum frequency used in vectorization')
    args = parser.parse_args()

    df = pd.read_json(f'../datasets/preprocessed/poa-{args.dataset}.json')

    # filters
    df = df[df['type'].str.contains('Apartamento')]
    # select only those with descriptions
    df = df[df['description'] != ' ']

    relevant_features.append('description_no_stp')
    df = df[relevant_features] if not args.text_only else df[[
        'description_no_stp', 'price']]

    # build textual pipeline
    textual_pipeline = create_text_pipe(args)
    preprocess = ColumnTransformer(
        [('bow', textual_pipeline, 'description_no_stp')],
        remainder='passthrough',
    )

    print(f'Textual Preprocess Pipeline:\n{preprocess}')

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
            ('preprocess', preprocess),
            ('regressor', regressor),
        ])

        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        X, y = df.drop(columns='price'), df['price']

        # print(X.iloc[0])
        # print(pipeline.fit_transform(X)[0])

        # break

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
            np.sqrt(results['test_mse'].mean()),
            results['test_mape'].mean(),
            results['test_mdape'].mean(),
            results['test_r2'].mean(),
        ))
