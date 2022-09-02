import pandas as pd
import numpy as np
import argparse
from sklearn.decomposition import TruncatedSVD
from utils import median_absolute_percentage_error, relevant_features, LoadPreComputedEmbeddings, WordEmbeddingsTransformer, BERTTransformer
from sklearn.model_selection import KFold, cross_validate
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import make_scorer, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer


def create_text_pipe(args):
    if args.pre_computed:
        embeddings = LoadPreComputedEmbeddings(
            args.dataset, args.embedding, args.desc, args.fine_tune)
    else:
        embeddings = WordEmbeddingsTransformer(
        ) if args.embedding == 'word' else BERTTransformer()

    if args.n_components > 0:
        return make_pipeline(embeddings, TruncatedSVD(n_components=args.n_components, random_state=42))
    else:
        return make_pipeline(embeddings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str,
                        choices=['rent', 'sale', 'homes'], required=True, help='available datasets are rent and sale')
    parser.add_argument('-e', '--embedding', choices=['word', 'bert'], required=True,
                        help='embedding mode to use')
    parser.add_argument('--desc', default='description_preprocess', type=str)
    parser.add_argument('-t', '--text_only', action='store_true',
                        help='if set, the model will use only textual information')
    parser.add_argument('-n', '--n_components', type=int, default=0,
                        help='n_components > 0 will apply dimensionality reduction with TruncatedSVD using n_components')
    parser.add_argument('-pc', '--pre_computed', action='store_true',
                        help='if will use pre-computed embeddings')
    parser.add_argument('-ft', '--fine_tune', action='store_true', 
                        help='When to use fine-tuned embeddings')
    args = parser.parse_args()


    if args.dataset == 'homes':
        df = pd.read_json(f'../datasets/preprocessed/homes.json')
        print('Dataset has {:d} samples.'.format(len(df)))
        df = df[df['type'] != 'lots land']
        print(f'Dataframe has now {len(df)} samples')
        df = df.dropna()
        print(f'Dataframe has now {len(df)} samples')

        if args.text_only:
            df = df[[
                args.desc,
                'price',
            ]]
        else:
            df = df[[
                'area',
                'rooms',
                'bathrooms',
                'age',
                #'lat',
                #'lng',
                args.desc,
                'price',
            ]]
    else:
        df = pd.read_json(f'../datasets/preprocessed/poa-{args.dataset}.json')

        # filters
        df = df[df['type'].str.contains('Apartamento')]
        # select only those with descriptions
        df = df[df['description'] != ' ']

        relevant_features.append(args.desc)
        df = df[relevant_features] if not args.text_only else df[[
            args.desc, 'price']]

    # document embeddings
    textual_pipeline = create_text_pipe(args)
    preprocess = ColumnTransformer(
        [('bow', textual_pipeline, args.desc)],
        remainder='passthrough')

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
        X, y = df.drop(columns=['price']), df['price']

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
