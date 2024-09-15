import json
import argparse
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from joblib import dump, load


def preprocessing():
    '''
    brief: This function preprocesses the data by loading it, 
           dropping unnecessary columns, converting data types,
           sorting by date, and resetting the index
    return: df (DataFrame) - Preprocessed DataFrame
    '''
    df = load(args.db)
    df.reset_index(drop=True, inplace=True)
    df.duplicated().sum()
    df.isnull().sum()

    df.drop(['Adj Close', 'Volume'], axis=1, inplace=True)

    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', ascending=True, inplace=True)

    return df


def transform_data(df):
    '''
    param: df (DataFrame) - DataFrame containing the data to be transformed
    return: X (array) - Transformed feature array
    '''
    features = ["Open", "High", "Low"]

    transformer = ColumnTransformer(
        transformers=[
            ('features', make_pipeline(
                SimpleImputer(missing_values=np.nan, strategy='median'),
                StandardScaler()
            ), features)
        ],
        remainder='passthrough'
    )

    X = transformer.fit_transform(df[features])

    dump(transformer, args.transformer)
    return X

def load_data(args):
    '''
    brief: Perform data preprocessing, feature transformation, split the data into train and test sets, 
            and save the processed data into a JSON file for training of Random Forest Regressor and Linear Regressor.
    param: args object containing command-line arguments
    
    '''
    df = preprocessing()
    X = transform_data(df)
    y = df['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)

    X_train, X_test, y_train, y_test = X_train.tolist(), X_test.tolist(), y_train.tolist(), y_test.tolist()

    data = {'x_train': X_train, 'y_train': y_train, 'x_test': X_test, 'y_test': y_test}

    data_json = json.dumps(data, indent=4)

    with open(args.data, 'w') as out_file:
        out_file.write(data_json)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, help="Insert path to get database")
    parser.add_argument('--data', type=str, help="Insert path to save data")
    parser.add_argument('--transformer', type=str, help="Insert path to save data")
    args = parser.parse_args()

    Path(args.data).parent.mkdir(parents=True, exist_ok=True)
    Path(args.transformer).parent.mkdir(parents=True, exist_ok=True)
    load_data(args)
