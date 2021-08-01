# Global import
import numpy as np
import os
from random import choice
import string
from pathlib import Path
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import StringType
from pyspark.sql.types import FloatType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import BooleanType

# Local import
from datalab.format_features import format_features
from datalab.fit_random_forest import tree_ditribution

l_random_string = [''.join(choice(string.ascii_letters) for _ in range(10)) for _ in range(100)]

schema = StructType([
    StructField('Field 1', IntegerType(), nullable=True),
    StructField('Field 2', IntegerType(), nullable=True),
    StructField('Field 3', FloatType(), nullable=True),
    StructField('Field 4', FloatType(), nullable=True),
    StructField('Field 5', StringType(), nullable=True),
    StructField('Label', BooleanType(), nullable=True),
])


def main_test():

    # Create conf and spark context
    conf = (
        SparkConf()
        .setAppName(os.getenv("APP_NAME"))
        .setMaster(os.getenv("SPARK_MASTER"))
    )
    sc = SparkContext(conf=conf)
    sc.addPyFile("dist/datalab-1.0-py3.7.egg")
    spark = SparkSession(sc)

    # Create fake dataframe

    l_randoms = [
        {
            "Field 1": np.random.randint(0, 1000), "Field 2": np.random.randint(0, 1000), "Field 3": np.random.randn(),
            "Field 4": np.random.randn(), "Field 5": choice(l_random_string), "Label": bool(np.random.binomial(1, 0.5))
        }
        for _ in range(10000)
    ]

    dfs_random = sc.parallelize(l_randoms).toDF(schema=schema)
    dfs_random.show()

    # Format data in dataframe
    pipeline, dfs_formatted = format_features(
        dfs_random, l_cols_num=['Field 1', 'Field 2', 'Field 3', 'Field 4'], l_cols_to_index=['Field 5'])
    dfs_formatted.show()

    # Fit random forest
    l_trees = tree_ditribution(dfs_formatted, 0.3, 10, sc)


if __name__ == "__main__":

    # Get project path and load (if any) local env
    project_path = Path(os.path.join(os.getcwd().split('heka')[0]))
    data_dir = project_path / 'data'

    # Set local params
    local_env = {
        'DATA_DIR': str(data_dir), "SPARK_MASTER": "spark://127.0.0.1:7077",
        "APP_NAME": "af_use_case"
    }
    param_env = {}

    # Update env variables without overwriting existing variables
    os.environ.update({k: os.environ.get(k, v) for k, v in list(local_env.items()) + list(param_env.items())})

    main_test()

