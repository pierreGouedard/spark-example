import wget
import os
import sys
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
import numpy as np
import math


def first_step_rdd():

    conf = (
        SparkConf()
        .setAppName(os.getenv("APP_NAME"))
        .setMaster(os.getenv("SPARK_MASTER"))

    )
    sc = SparkContext(conf=conf)

    # Create fake data
    l_data = [
        (np.random.randint(0, 10), np.random.randn()) for i in range(int(os.getenv('N_VALUE')))
    ]

    # Create rdd and implement simple pipeline
    rdd_data = sc.parallelize(l_data)
    rdd_sum = rdd_data.groupByKey()\
        .map(lambda x: (x[0], sum(x[1]) / len(x[1])))

    res = rdd_sum.collect()
    print(res)
    sc.stop()


if __name__ == "__main__":
    # Get project path and load (if any) local env
    project_path = Path(os.path.join(os.getcwd().split('heka')[0]))
    data_dir = project_path / 'data'

    # Set local params
    local_env = {'DATA_DIR': str(data_dir), "SPARK_MASTER": "spark://127.0.0.1:7077", "APP_NAME": "first-step"}
    param_env = {'N_VALUE': "1000000"}

    # Update env variables without overwriting existing variables
    os.environ.update({k: os.environ.get(k, v) for k, v in list(local_env.items()) + list(param_env.items())})

    first_step_rdd()
