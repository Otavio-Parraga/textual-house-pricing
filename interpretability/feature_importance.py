import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import eli5
import json
from eli5.sklearn import PermutationImportance
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error, mean_absolute_error
from nltk.corpus import stopwords


# metrics
def median_absolute_percentage_error(y_true, y_pred):
    return np.median(np.abs((y_true - y_pred) / y_true))


def create_text_pipe(args):
    STOPWORDS = set(stopwords.words('portuguese')) if args.dataset != 'homes' else set(stopwords.words('english'))
    counter = CountVectorizer(
        binary=args.binary, stop_words=STOPWORDS, min_df=args.min_df, dtype=np.float32)
    if args.tfidf:
        return make_pipeline(counter, TfidfTransformer())
    return make_pipeline(counter)


def get_model(args):
    if args.model == 'lgbm':
        return LGBMRegressor(random_state=42)
    elif args.model == 'xgboost':
        import xgboost
        if '1.3' <= xgboost.__version__ < '1.4':
            return xgboost.XGBRegressor(random_state=42)
        else:
            raise ValueError('xgboost version must be between 1.3 and 1.3.3')
    elif args.model == 'rf':
        return RandomForestRegressor(random_state=42)
    elif args.model == 'et':
        return ExtraTreesRegressor(random_state=42)
    elif args.model == 'gbm':
        return GradientBoostingRegressor(random_state=42)
    elif args.model == 'lr':
        return LinearRegression()
    elif args.model == 'dt':
        return DecisionTreeRegressor(random_state=42)
    else:
        raise ValueError('Invalid model')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str,
                        choices=['rent', 'sale', 'homes'], required=True, help='available datasets are rent and sale')
    parser.add_argument('-b', '--binary', action='store_true', help='make count vectorization binary')
    parser.add_argument('-tf', '--tfidf', action='store_true', help='apply tfidf after vectorization')
    parser.add_argument('-md', '--min_df', type=float, default=0.001, help='minimum frequency used in vectorization')
    parser.add_argument('-m', '--model', type=str, choices=['lgbm', 'xgboost', 'rf', 'et', 'gbm', 'lr', 'dt'],
                        required=True)
    parser.add_argument('-i', '--instances', type=int, nargs='+', default=[])
    parser.add_argument('-perm', '--permutation', action='store_true', help='run permutation importance')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    out_dir = Path(f'./{args.dataset}/{args.model}/binary-{args.binary}/tfidf-{args.tfidf}/min_df-{args.min_df}')
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ['rent', 'sale']:
        df = pd.read_json(f'../datasets/preprocessed/poa-{args.dataset}.json')
        df = df[df['type'].str.contains('Apartamento')]
        df = df[df['description'] != ' ']
    else:
        df = pd.read_json(f'../datasets/preprocessed/homes.json')
        df = df[df['type'] != 'lots land']
        df = df.dropna()
    
    df = df[['description_preprocess','price']]

    # build textual pipeline
    textual_pipeline = create_text_pipe(args)
    preprocess = ColumnTransformer(
        [('bow', textual_pipeline, 'description_preprocess')],
        remainder='passthrough',
    )

    model = get_model(args)

    pipeline = Pipeline(steps=[
        ('preprocess', preprocess),
        ('regressor', model),
    ])

    X, y = df.drop(columns='price'), df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mdape = median_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    print('MSE: {:.3f} | MAE: {:.3f} | RMSE: {:.3f} | MAPE: {:.3f} | MdAPE: {:.3f} | R2: {:.3f}'.format(
        mse, mae, rmse, mape, mdape, r2))

    metrics = {'metrics': {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'MdAPE': mdape,
        'R2': r2,
    }}

    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f)

    count_vectorizer = pipeline.named_steps['preprocess'].transformers_[0][1].named_steps['countvectorizer']

    # use model weights to get the most important features
    explainer = eli5.explain_weights(pipeline['regressor'], vec=count_vectorizer, top=None)
    eli5.formatters.as_dataframe.format_as_dataframe(explainer).to_csv(out_dir / 'weights.csv', index=False)

    if len(args.instances) > 0:
        for i in args.instances:
            explainer = eli5.explain_prediction(pipeline['regressor'], X_test.iloc[i][0], vec=count_vectorizer, top=None)
            with open(f'{out_dir}/instance-{i}.html', 'w') as f:
                f.write(eli5.formatters.format_as_html(explainer))

    # use permutation importance to get the most important features
    if args.permutation:
        X_test = pipeline.named_steps['preprocess'].transform(X_test)
        perm_imp = PermutationImportance(pipeline.named_steps['regressor'], random_state=42).fit(X_test.toarray(), y_test)
        explainer = eli5.explain_weights(perm_imp, vec=count_vectorizer, top=None)
        eli5.formatters.as_dataframe.format_as_dataframe(explainer).to_csv(out_dir / 'perm_imp.csv', index=False)
