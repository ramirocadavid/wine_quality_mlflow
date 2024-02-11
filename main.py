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

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

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

    # Read the wine-quality csv file from local
    data = pd.read_csv("data/red-wine-quality.csv")

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = args.alpha
    l1_ratio = args.l1_ratio
    
    # Define folder for this project
    mlflow.set_tracking_uri("./wine_quaility_mlflow")
    print(f"Tracking URI is {mlflow.get_tracking_uri()}")

    # Define and run experiment
    exp_id = mlflow.create_experiment(
        name="naive_regressor_3",
        tags={"version": "v1", "priority": "p1"}
    )
    exp = mlflow.get_experiment(exp_id)
    
    print(f"Name: {exp.name}")
    print(f"ID: {exp.experiment_id}")
    print(f"Artifact location: {exp.artifact_location}")
    print(f"Tags: {exp.tags}")
    print(f"Lifecycle stage: {exp.lifecycle_stage}")
    print(f"Creation timestamp: {exp.creation_time}")


    with mlflow.start_run(experiment_id=exp_id):
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
        
        # Log parameters
        mlflow.log_params({
            'alpha': alpha,
            'l1_ratio': l1_ratio
        })
        # Log metrics
        mlflow.log_metrics(
            metrics = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            },
            step=0
        )
        # Log model
        mlflow.sklearn.log_model(lr, "lr_model")
