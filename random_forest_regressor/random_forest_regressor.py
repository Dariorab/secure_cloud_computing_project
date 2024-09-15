import json, os
from joblib import dump
import argparse
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor # mettere RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np


def random_forest_regressor(args):
    '''
    brief: This function trains a Random Forest Regressor model using the provided data, 
            performs hyperparameter tuning using GridSearchCV,
            saves the best model, evaluates the model on the test data, 
            and writes the R2 score to a file.
    params: args object containing command-line arguments
    '''
    
    with open(args.data) as data_file:
        data = json.load(data_file)

    x_train, x_test, y_train, y_test = data['x_train'], data['x_test'], data['y_train'], data['y_test']

    parameter_grid={'n_estimators':[64, 128, 256],
                    'max_depth':[2, 4, 8, 16, 36, 64]}

    model = RandomForestRegressor(random_state=42)

    # Create GridSearchCV
    grid_search = GridSearchCV(model, parameter_grid, n_jobs=-1, scoring='r2', cv=5)

    grid_search.fit(x_train, y_train)

    model = grid_search.best_estimator_
    dump(model, args.model)

    y_pred = model.predict(x_test)

    y_pred = model.predict(x_test)
    
    y_pred = y_pred.reshape(-1,1)
    y_test = np.array(y_test).reshape(-1,1)

    y_pred = StandardScaler().fit_transform(y_pred)
    y_test = StandardScaler().fit_transform(y_test)
    score = r2_score(y_test, y_pred)

    with open(args.r2, 'w') as score_file:
        score_file.write(str(score))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Forest Regressor')
    parser.add_argument('--data', type=str)
    parser.add_argument('--r2', type=str)
    parser.add_argument("--model", type=str)

    args = parser.parse_args()

    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    Path(args.r2).parent.mkdir(parents=True, exist_ok=True)

    random_forest_regressor(args)