import kfp
from kfp import dsl
from kfp.components import func_to_container_op
from kfp.compiler.compiler import Compiler

@func_to_container_op
def show_results(random_forest: float, linear_regression: float, lstm: float) -> None:
    '''
    brief: Given the outputs from decision_tree and logistic regression components
           the results are shown.
    '''
    print(f"Random Forest Regression (R-squared): {random_forest}")
    print(f"Linear regression (R-squared): {linear_regression}")
    print(f"LSTM (R-squared): {lstm}")


@dsl.pipeline(name='ETH Pipeline', description='Applies Random Forest Classifier and Linear Regression for ETH classification problem.')
def eth_pipeline():

    '''
    brief: creation of Pipeline to apply Random Forest Regressor, Linear Regression, and LSTM models 
           for ETH classification problem.
    '''

    # Loads the yaml manifest for each component
    download = kfp.components.load_component_from_file('download_data/download_data.yaml')
    load = kfp.components.load_component_from_file('load_data/load_data.yaml')
    load_lstm = kfp.components.load_component_from_file('lstm_load_data/lstm_load_data.yaml')
    random_forest = kfp.components.load_component_from_file('random_forest_regressor/random_forest_regressor.yaml')
    linear_regression = kfp.components.load_component_from_file('linear_regression/linear_regression.yaml')
    lstm = kfp.components.load_component_from_file('lstm/lstm.yaml')

    # Run load_data task
    download_task = download()
    load_task = load(download_task.output)
    load_task_lstm = load_lstm(download_task.output)
    
    random_forest_task = random_forest(load_task.output)
    linear_regression_task = linear_regression(load_task.output)
    lstm_task = lstm(load_task_lstm.output)

    show_results(random_forest_task.outputs['R2'], linear_regression_task.outputs['R2'], lstm_task.outputs['R2'])


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(eth_pipeline, 'eth_pipeline.yaml')

