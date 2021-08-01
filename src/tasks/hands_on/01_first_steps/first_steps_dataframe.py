import wget
import os
import pandas as pd
from pathlib import Path
from pyspark.sql import SparkSession


def first_step_dataframe():
    # Get params and env variable
    data_dir = os.getenv('DATA_DIR')
    url_data = os.getenv('URL_DATA')

    # Init Spark session
    ss = (
        SparkSession.
        builder.
        appName(os.getenv("APP_NAME")).
        master(os.getenv("SPARK_MASTER")).
        config("spark.executor.memory", "512m").
        getOrCreate()
    )

    # Download data
    wget.download(url_data, out=data_dir)
    df_iris = pd.read_csv(os.path.join(data_dir, 'iris.data'))

    # Create dataframe and show
    dfs_iris = ss.createDataFrame(df_iris)
    dfs_iris.show(5)
    print(dfs_iris.rdd.map(lambda row: row.asDict()).collect()[:5])
    ss.stop()


if __name__ == "__main__":
    # Get project path and load (if any) local env
    project_path = Path(os.path.join(os.getcwd().split('heka')[0]))
    data_dir = project_path / 'data'

    # Set local params
    local_env = {'DATA_DIR': str(data_dir), "SPARK_MASTER": "spark://127.0.0.1:7077", "APP_NAME": "first-step-dataframe"}
    param_env = {'URL_DATA': "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"}

    # Update env variables without overwriting existing variables
    os.environ.update({k: os.environ.get(k, v) for k, v in list(local_env.items()) + list(param_env.items())})

    first_step_dataframe()
