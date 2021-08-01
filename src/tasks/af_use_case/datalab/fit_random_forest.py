import numpy as np
from sklearn import tree


def sample_row(n_tree, p_sampling):
    return [i * np.random.binomial(1, p_sampling) - 1 for i in range(1, n_tree + 1)]


def tree_ditribution(dfs, p_sampling, n_tree, sc):

    n_tree_bc, p_sampling_bc = sc.broadcast(n_tree), sc.broadcast(p_sampling)

    l_trees = dfs.rdd.map(
        lambda row: {"treeKeys": sample_row(n_tree_bc.value, p_sampling_bc.value), **row.asDict()}
    )\
        .flatMap(
        lambda x: [(k, {'Features': x['Features'], 'Label': x['Label']}) for k in x['treeKeys']]
    )\
        .groupByKey()\
        .map(lambda x: fit_random_forest([v['Features'] for v in x[1]], [v['Label'] for v in x[1]]))\
        .collect()

    return l_trees


def fit_random_forest(l_features, l_labels):
    X, y = np.stack(l_features), np.stack(l_labels)
    clf = tree.DecisionTreeClassifier().fit(X, y)
    return clf