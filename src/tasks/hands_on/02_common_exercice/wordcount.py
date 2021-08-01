
import os
import pandas as pd
from pathlib import Path
from pyspark import SparkConf, SparkContext
from operator import add


def wordcount():

    # Create conf and spark context
    conf = (
        SparkConf()
        .setAppName(os.getenv("APP_NAME"))
        .setMaster(os.getenv("SPARK_MASTER"))

    )
    sc = SparkContext(conf=conf)

    # load large text
    with open(os.path.join(os.getenv('DATA_DIR'), 'bible.txt'), 'r') as f:
        l_large_text = f.readlines()

    # wordcount using spark and functional logic
    lines = sc.parallelize(l_large_text)
    counts = lines.flatMap(lambda x: x.split(' ')) \
                  .map(lambda x: (x, 1)) \
                  .reduceByKey(add)
    counts.persist()
    
    l_tops = counts.map(lambda x:  x[1]) \
                  .sortBy(lambda x: x, False) \
                  .take(10)

    max_bc = sc.broadcast(l_tops[-1])

    top_counts = counts.filter(lambda x: x[1] >= max_bc.value) \
        .collect()

    # Display result
    for (word, count) in top_counts:
        print("%s: %i" % (word, count))

    sc.stop()


if __name__ == "__main__":
    # Get project path and load (if any) local env
    project_path = Path(os.path.join(os.getcwd().split('heka')[0]))
    data_dir = project_path / 'data'

    # Set local params
    local_env = {
        'DATA_DIR': str(data_dir), "SPARK_MASTER": "spark://127.0.0.1:7077",
        "APP_NAME": "word-count"
    }
    param_env = {'URL_DATA': "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"}

    # Update env variables without overwriting existing variables
    os.environ.update({k: os.environ.get(k, v) for k, v in list(local_env.items()) + list(param_env.items())})

    wordcount()
