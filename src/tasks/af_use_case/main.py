# Global import
import os
from pathlib import Path
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

# Local import
from datalab.random_forest_selector import SparkRfClfSelector, SparkRfRegSelector
from datalab.data_generator import generate_data, parameters


def main_clf_test():

    # Create conf and spark context
    conf = (
        SparkConf()
        .setAppName(os.getenv("APP_NAME"))
        .setMaster(os.getenv("SPARK_MASTER"))
    )
    sc = SparkContext(conf=conf)
    sc.addPyFile("dist/datalab-1.0-py3.7.egg")
    spark = SparkSession(sc)

    # Generate data
    transformer, dfs_train, dfs_test = generate_data(sc)

    # Fit random forest
    rf_selector = SparkRfClfSelector(
        sc, 'local', parameters['param_grid'], parameters['n_trees'], parameters['max_p_sampling'],
        parameters['max_sample_by_tree'], parameters['col_p_sample']
    )

    # evaluate with test and get best model
    best_model = rf_selector.fit(dfs_train).evaluate(dfs_test).best_model
    dfs_preds = best_model.predict(dfs_test)

    dfs_preds.show(10)

    spark.stop()
    sc.stop()


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

    main_clf_test()
