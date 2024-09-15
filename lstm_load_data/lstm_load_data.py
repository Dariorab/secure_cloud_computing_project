import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import argparse, json
from pathlib import Path
from joblib import load

def preprocessing():
    '''
    brief: This function preprocesses the data by loading it, 
           dropping unnecessary columns, converting data types,
           sorting by date, and resetting the index
    return: df (DataFrame) - Preprocessed DataFrame
    '''

    df= load(args.data)
    
    # Close and Adj CLose are equals, so i can drop Adj Close
    df.drop(['Adj Close', 'Volume'], axis=1, inplace=True)

    df.info()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    NumCols = df.columns.drop(['Date'])
    df[NumCols] = df[NumCols].replace({',': ''}, regex=True)
    df[NumCols] = df[NumCols].astype('float64')

    return df

def transform_data(df):
    '''
    param: df (DataFrame) - DataFrame containing the data to be transformed
    return: scaler (MinMaxScaler) - Fitted MinMaxScaler object for scaling data
    '''

    scaler = MinMaxScaler()
    scaler.fit(df.Close.values.reshape(-1, 1))

    from joblib import dump
    dump(scaler, "scaler.joblib")

    return scaler

def load_data(args):
    '''
    brief: Load data, preprocess it, transform it using MinMaxScaler,
        split it into train and test sets, 
        and save the processed data into a JSON file 
        as required for training of LSTM model.
    param: args object containing command-line arguments
    '''
    window_size = 60

    df = preprocessing()
    scaler = transform_data(df)
    test_size = df[(df.Date.dt.year==2023) | (df.Date.dt.year==2024)].shape[0]

    # Train set
    train_data = df.Close[:-test_size]
    train_data = scaler.transform(train_data.values.reshape(-1,1))

    X_train = []
    y_train = []

    for i in range(window_size, len(train_data)):
        X_train.append(train_data[i-window_size:i, 0])
        y_train.append(train_data[i, 0])

    # Test set
    test_data = df.Close[-test_size-window_size:]
    test_data = scaler.transform(test_data.values.reshape(-1,1))

    X_test = []
    y_test = []

    for i in range(window_size, len(test_data)):
        X_test.append(test_data[i-window_size:i, 0])
        y_test.append(test_data[i, 0])
    
    X_train = np.array(X_train)
    X_test  = np.array(X_test)
    y_train = np.array(y_train)
    y_test  = np.array(y_test)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test  = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_train = np.reshape(y_train, (-1,1))
    y_test  = np.reshape(y_test, (-1,1))
    
    data = {'x_train': X_train.tolist(), 'y_train': y_train.tolist(), 'x_test': X_test.tolist(), 'y_test': y_test.tolist()}

    data_json = json.dumps(data, indent=4)

    with open(args.data, 'w') as out_file:
        out_file.write(data_json)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, help="Insert path to get database")
    parser.add_argument('--data', type=str, help="Insert path to save data")
    args = parser.parse_args()

    Path(args.data).parent.mkdir(parents=True, exist_ok=True)
    
    load_data(args)