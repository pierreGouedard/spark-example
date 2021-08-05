# Global import
import numpy as np
from random import choice
import string
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.pipeline import Pipeline
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import StringType
from pyspark.sql.types import FloatType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import BooleanType

l_random_string = [''.join(choice(string.ascii_letters) for _ in range(10)) for _ in range(100)]

schema = StructType([
    StructField('Field 1', IntegerType(), nullable=True),
    StructField('Field 2', IntegerType(), nullable=True),
    StructField('Field 3', FloatType(), nullable=True),
    StructField('Field 4', FloatType(), nullable=True),
    StructField('Field 5', StringType(), nullable=True),
    StructField('Label', BooleanType(), nullable=True),
])

parameters = {
    "param_grid": {"min_samples_leaf": [100, 200], "max_depth": [5, 10]}, 'clf_metric': "areaUnderRoc",
    "reg_metrics": "mse", "n_trees": 100, "max_p_sampling": 0.5, "max_sample_by_tree": 500000, "col_p_sample": 0.7
}


def generate_data(sc):

    # Create fake train and test data
    dfs_train = sc.parallelize(random_values(1000000)).toDF(schema=schema)
    dfs_test = sc.parallelize(random_values(50000)).toDF(schema=schema)

    # Format data in dataframe
    transformer, dfs_train = format_features(
        dfs_train, l_cols_num=['Field 1', 'Field 2', 'Field 3', 'Field 4'], l_cols_to_index=['Field 5']
    )

    dfs_test = transformer.transform(dfs_test)

    return transformer, dfs_train, dfs_test


def random_values(n):
    return [
        {
            "Field 1": np.random.randint(0, 1000), "Field 2": np.random.randint(0, 1000), "Field 3": np.random.randn(),
            "Field 4": np.random.randn(), "Field 5": choice(l_random_string), "Label": bool(np.random.binomial(1, 0.5))
        }
        for _ in range(n)
    ]


def format_features(dfs, l_cols_num, l_cols_to_index):

    pipeline = Pipeline(stages=[
        *[StringIndexer(inputCol=c, outputCol=f'{c} (encoded)') for c in l_cols_to_index],
        VectorAssembler(inputCols=l_cols_num + [' '.join([c, '(encoded)']) for c in l_cols_to_index], outputCol='Features')
    ])

    transformer = pipeline.fit(dfs)
    dfs = transformer.transform(dfs)

    dfs = dfs.drop(*[c for c in dfs.columns if c not in ['Features', 'Label']])

    return transformer, dfs
