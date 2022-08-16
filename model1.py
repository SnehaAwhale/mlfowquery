import os
import argparse
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
# from streamlit import c


# mlflow.set_tracking_uri('file:///C:/Users/eawhsne/PycharmProjects/mlflowSunny/mlruns')
# client=mlflow.tracking.MlflowClient('file:///C:/Users/eawhsne/PycharmProjects/mlflowSunny/mlruns')

# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# client=mlflow.tracking.MlflowClient("http://127.0.0.1:5000")
mlflow.set_tracking_uri('sqlite:///mlflow.db')
# client=mlflow.tracking.MlflowClient('sqlite:///mlflow.db')

experiments='exp15'
# client.create_experiment(name=experiments,artifact_location=experiments)
mlflow.set_experiment(experiment_name=experiments)
# print('tracking uri is :',mlflow.get_tracking_uri())



#
# experiments='exp11'
# client.create_experiment(name=experiments,artifact_location=experiments)

# mlflow.delete_experiment(3)




def get_data():
    URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

    try:
        df = pd.read_csv(URL, sep=";")
        return df
    except Exception as e:
        raise e

def evaluate(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def main(alpha, l1_ratio):

    df = get_data()

    train, test = train_test_split(df)

    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)

    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # experiment_name='exp14'
    # experiment_ids=client.get_experiment_by_name(experiment_name).experiment_id
    # print('exp id is',experiment_ids)
    with mlflow.start_run():
        print('ye he active run :', mlflow.active_run())
        print('artifacts uri is :', mlflow.get_artifact_uri())
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        pickle_out=open("classifier.pkl","wb")
        pickle.dump(lr,pickle_out)
        pickle_out.close()

        pred = lr.predict(test_x)

        rmse, mae, r2 = evaluate(test_y, pred)

        print(f"Elastic net params: alpha: {alpha}, l1_ratio: {l1_ratio}")
        print(f"Elastic net metric: rmse:{rmse}, mae: {mae}, r2:{r2}")

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_artifact("classifier.pkl",artifact_path='classifier')
        mlflow.sklearn.log_model(lr, "model")

    mlflow.end_run()

# main(0.7,0.7)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--alpha", "-a", type=float, default=0.5)
    args.add_argument("--l1_ratio", "-l1", type=float, default=0.5)
    parsed_args = args.parse_args()
    try:
        main(alpha=parsed_args.alpha, l1_ratio=parsed_args.l1_ratio)
    except Exception as e:
        raise e