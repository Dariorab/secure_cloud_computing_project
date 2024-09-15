import json
from joblib import dump
import argparse
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


def linear_regression(args):
    '''
    
    brief: Load preprocessed data, perform hyperparameter tuning for Linear Regression model using GridSearchCV, 
            save the best model, evaluate it on the test set, and write the R2 score to a file.
    param: args (Namespace) - Namespace object containing command-line arguments
    '''

    data = None
    with open(args.data) as data_file:
        data = json.load(data_file)

    x_train, x_test, y_train, y_test = data['x_train'], data['x_test'], data['y_train'], data['y_test']

    param_grid={'fit_intercept':(True, False), 'copy_X':(True, False)}

    model = LinearRegression()
    grid_search = GridSearchCV(model, param_grid, n_jobs=-1, scoring='r2', cv=5)

    grid_search.fit(x_train, y_train)

    model = grid_search.best_estimator_

    dump(model, args.model)

    y_pred = model.predict(x_test)
    
    y_pred = y_pred.reshape(-1,1)
    y_test = np.array(y_test).reshape(-1,1)

    y_pred = StandardScaler().fit_transform(y_pred)
    y_test = StandardScaler().fit_transform(y_test)
    score = r2_score(y_test, y_pred)

    with open(args.r2, 'w') as score_file:
        score_file.write(str(score))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Linear Regression')
    parser.add_argument('--data', type=str)
    parser.add_argument('--r2', type=str)
    parser.add_argument('--model', type=str)

    args = parser.parse_args()

    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    Path(args.r2).parent.mkdir(parents=True, exist_ok=True)

    linear_regression(args)
