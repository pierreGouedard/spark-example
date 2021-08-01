import wget
import os
import pandas as pd
from pathlib import Path
from pyspark import SparkConf, SparkContext
from random import random

def is_inside(p):
    # p is useless here
    x, y = random(), random()
    return 1 if x*x + y*y < 1 else 0

def compute_pi():

    # Create conf and spark context
    conf = (
        SparkConf()
        .setAppName(os.getenv("APP_NAME"))
        .setMaster(os.getenv("SPARK_MASTER"))

    )
    sc = SparkContext(conf=conf)

    # Estimate PI
    n = int(os.getenv('N_VALUE'))
    count = sc.parallelize(range(0, n)) \
             .map(is_inside) \
             .reduce(lambda x, y: x + y)

    print("Pi is roughly %f" % (4.0 * count / n))
    sc.stop()


if __name__ == "__main__":
    # Get project path and load (if any) local env
    project_path = Path(os.path.join(os.getcwd().split('heka')[0]))
    data_dir = project_path / 'data'

    # Set local params
    local_env = {
        'DATA_DIR': str(data_dir), "SPARK_MASTER": "spark://127.0.0.1:7077",
        "APP_NAME": "compute-pi"
    }
    param_env = {'N_VALUE': "100000"}

    # Update env variables without overwriting existing variables
    os.environ.update({k: os.environ.get(k, v) for k, v in list(local_env.items()) + list(param_env.items())})

    compute_pi()
