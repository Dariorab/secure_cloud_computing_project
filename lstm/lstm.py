import numpy as np
import pandas as pd
import argparse, json
from sklearn.metrics import r2_score
import tensorflow as tf
from keras import Model
from keras.layers import Input, Dense, Dropout
from keras.layers import LSTM
from pathlib import Path
from joblib import dump

def lstm():
    '''
    brief: define LSTM neural network model architecture for time series prediction.
    return: instance of LSTM
    '''
    window_size = 60
    input1 = Input(shape=(window_size,1))
    x = LSTM(units = 64, return_sequences=True)(input1)
    x = Dropout(0.2)(x)

    x = LSTM(units = 64, return_sequences=True)(x)
    x = Dropout(0.2)(x)

    x = LSTM(units = 64)(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='linear')(x)
    dnn_output = Dense(30)(x)

    model = Model(inputs=input1, outputs=[dnn_output])
    model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['r2_score'])
    model.summary()

    return model

def train_lstm(args):
    '''
    brief: Load preprocessed data, train an LSTM model, evaluate it on the test set, 
            save the trained model, and write the R2 score to a file.
    param: args object containing command-line arguments
    '''
    data = None

    with open(args.data) as input_file:
        data = json.load(input_file)

    X_train = np.array(data['x_train'])
    y_train = np.array(data['y_train'])
    X_test = np.array(data['x_test'])
    y_test = np.array(data['y_test'])

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_train = np.reshape(y_train, (-1, 1))
    y_test = np.reshape(y_test, (-1, 1))

    model = lstm()
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
    dump(model, args.model)
    
    y_pred = model.predict(X_test)

    r2_score_lstm = r2_score(y_test, y_pred)

    with open(f'{args.r2}', 'w') as score_file:
        score_file.write(str(r2_score_lstm))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM')
    parser.add_argument('--data', type=str)
    parser.add_argument('--r2', type=str)
    parser.add_argument("--model", type=str)

    args = parser.parse_args()
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    Path(args.r2).parent.mkdir(parents=True, exist_ok=True)

    train_lstm(args)