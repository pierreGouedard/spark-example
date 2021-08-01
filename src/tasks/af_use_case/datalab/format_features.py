from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.pipeline import Pipeline


def format_features(dfs, l_cols_num, l_cols_to_index):

    pipeline = Pipeline(stages=[
        *[StringIndexer(inputCol=c, outputCol=f'{c} (encoded)') for c in l_cols_to_index],
        VectorAssembler(inputCols=l_cols_num + [' '.join([c, '(encoded)']) for c in l_cols_to_index], outputCol='Features')
    ])

    dfs = pipeline.fit(dfs).transform(dfs)
    dfs = dfs.drop(*[c for c in dfs.columns if c not in ['Features', 'Label']])

    return pipeline, dfs

