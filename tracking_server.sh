#! /bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mlflow
mlflow server --backend-store-uri sqlite:///mlflow-wine.db \
              --default-artifact-root ./mlflow-artifacts \
              --host 127.0.0.1 \
              --port 5000