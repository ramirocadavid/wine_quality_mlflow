import os
import warnings
import logging
# Data processing packages 
import pandas as pd
import numpy as np
# ML packages
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
# mlflow
import mlflow
mlflow.sklearn.autolog(log_input_examples=True, max_tuning_runs=20)

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Create project folders
if not os.path.isdir('data'):
    os.mkdir('data')

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    ########### Define and log experiments data ###########

    # Read the wine-quality csv file from local
    data = pd.read_csv("data/red-wine-quality.csv")

    # Split the data into training and test sets. (0.8, 0.2) split.
    train_x, test_x, train_y, test_y = train_test_split(
        data.drop(columns='quality'),
        data['quality'], 
        test_size=0.8
    )

    # Store data
    train_x.to_csv('./data/train_x.csv')
    test_x.to_csv('./data/test_x.csv')
    train_y.to_csv('./data/train_y.csv')
    test_y.to_csv('./data/test_y.csv')

    # Log data in a run
    exp_artifacts = mlflow.set_experiment(experiment_name='common-artifacts')
    with mlflow.start_run(run_name='common-datasets', 
                          experiment_id=exp_artifacts.experiment_id):
        mlflow.log_artifacts('./data/')

    # Define folder for this project
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    print(f"Tracking URI is {mlflow.get_tracking_uri()}")

    ########### Run experiments ###########

    # Three experiments will be run:
    #   - Lasso regression (l1_ratio=1)
    #   - Ridge regression (l1_ratio=0)
    #   - Elasticnet as mean of Lasso and Ridge (l1_ration=0.5)
    # Within each experiment, each run will test a different value of the
    #   regularization term (C).

    # Define and run experiment
    exp = mlflow.set_experiment(experiment_name=f"grid-search-cv")
    
    print(f"Experiment name: {exp.name}")
    print(f"ID: {exp.experiment_id}")
    print(f"Tags: {exp.tags}")
    print(f"Creation timestamp: {exp.creation_time}")
    
    with mlflow.start_run(
        experiment_id=exp.experiment_id, 
    ):
        # Define pipeline
        pipeline = Pipeline([
            ('regressor', ElasticNet())
        ])
        
        # Define grid search cv estimator 
        metrics = ['neg_mean_squared_error', 'r2', 'neg_mean_absolute_error']
        grid_params = [
            {
                'regressor': [ElasticNet()],
                'regressor__alpha': [0.001, 0.01, 0.1, 1],
                'regressor__l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0],
            },
            {
                'regressor': [RandomForestRegressor()],
                'regressor__n_estimators': [50, 100, 300]
            }
        ]
        gs_cv = GridSearchCV(pipeline, 
                             grid_params, 
                             scoring=metrics, 
                             refit='neg_mean_squared_error',
                             verbose=2)

        # Train the model
        gs_cv.fit(train_x, train_y)

        # Evaluate model and print results
        predicted_qualities = gs_cv.predict(test_x)

        # Log other data
        mlflow.log_artifact('data/red-wine-quality.csv')
        mlflow.set_tags({'run.type': 'prototype'})
    print("")
