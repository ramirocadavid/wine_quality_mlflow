import os
import warnings
import argparse
import logging
# Data processing packages 
import pandas as pd
import numpy as np
# ML packages
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
# mlflow
import mlflow
mlflow.autolog(log_input_examples=True)

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Create project folders
if not os.path.isdir('data'):
    os.mkdir('data')

# Get arguments from command
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.5)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.5)
args = parser.parse_args()

# Evaluation function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    ########### Define and log experiments data ###########

    # Read the wine-quality csv file from local
    data = pd.read_csv("data/red-wine-quality.csv")

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(columns=["quality"])
    test_x = test.drop(columns=["quality"])
    train_y = train[["quality"]]
    test_y = test[["quality"]]
    
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

    # Parse argument values
    alpha = args.alpha
    l1_ratio = args.l1_ratio
    
    # Define folder for this project
    mlflow.set_tracking_uri("")
    print(f"Tracking URI is {mlflow.get_tracking_uri()}")

    ########### Run experiments ###########

    # Three experiments will be run:
    #   - Lasso regression (l1_ratio=1)
    #   - Ridge regression (l1_ratio=0)
    #   - Elasticnet as mean of Lasso and Ridge (l1_ration=0.5)
    # Within each experiment, each run will test a different value of the
    #   regularization term (C).

    # Define parameters of experiments and runs
    exp_l1_ratio = [0, 0.5, 1]
    run_alpha = [0.001, 0.01, 0.1, 1]

    for l1_ratio in exp_l1_ratio:


        # Define and run experiment
        exp = mlflow.set_experiment(experiment_name=f"penalty_l{l1_ratio}")
        
        print(f"Experiment name: {exp.name}")
        print(f"ID: {exp.experiment_id}")
        print(f"Artifact location: {exp.artifact_location}")
        print(f"Tags: {exp.tags}")
        print(f"Lifecycle stage: {exp.lifecycle_stage}")
        print(f"Creation timestamp: {exp.creation_time}")
        
        for alpha in run_alpha:
            with mlflow.start_run(
                experiment_id=exp.experiment_id, 
                run_name=f"run_{alpha}"
            ):
                # Train the model
                lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
                lr.fit(train_x, train_y)

                # Evaluate model and print results
                predicted_qualities = lr.predict(test_x)
                (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
                print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
                print("  RMSE: %s" % rmse)
                print("  MAE: %s" % mae)
                print("  R2: %s" % r2)

                # Log other data
                mlflow.log_artifact('data/red-wine-quality.csv')
                mlflow.set_tags({'run.type': 'prototype'})
            print("")
        print("")
