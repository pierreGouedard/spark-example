# Global import
import itertools
import math
import numpy as np
from sklearn import tree
from sklearn import metrics

# Local import
from datalab.random_forest_model import DecisionTreeModel, RandomForestModel


class SparkRfSelector:
    """
    Evaluate and Select random forest parameters.

    This class has abstract methods that need to be implemented in child class.
    """
    allowed_params = {'min_samples_leaf': int, 'max_depth': int}
    min_col_sampled = 5

    def __init__(
            self, sc, model_path, param_grid, n_trees, max_p_sampling, max_sample_by_tree=750000,
            col_p_sample=0.8
    ):
        """
        Constructor.

        Parameters
        ----------
        sc: Sparkontext:
        model_path: str:
        param_grid: dict:
            Grid of Random forest params whose combination needs to be tested.
        n_trees: int:
            Number of tree in the Random forest.
        max_p_sampling: float:
            Maximum sampling rate used to sample input dataset to train a single tree.
        max_sample_by_tree: int:
            Maximum number of samples used to fit a single tree.
        col_p_sample: float:
            Sampling rate of the column of input dataset to train a single tree.

        """
        self.sc, self.model_path = sc, model_path

        # Parameter model
        self.param_grid, self.col_p_sample = param_grid, col_p_sample
        self.n_trees, self.max_p_sampling, self.max_sample_by_tree = n_trees, max_p_sampling, max_sample_by_tree
        self.param_keys = self.compute_param_keys(self.param_grid)

        # Fitted models
        self.models, self.best_model = None, None

    def fit(self, dfs_train):
        """
        Fit the random forest from a training dataset. (Abstract method)

        Parameters
        ----------
        dfs_train: pyspark.sql.DataFrame

        Returns
        -------
        'SparkRfSelector'

        """
        pass

    def evaluate(self, dfs_validation):
        """
        Evaluate the Random forest with a validation dataset. (Abstract method)

        Parameters
        ----------
        dfs_validation: pyspark.sql.DataFrame

        Returns
        -------
        None

        """
        pass

    def distribute_samples_for_train(self, dfs_train):
        """
        Distribute samples across trees compounding the random forest.

        Parameters
        ----------
        dfs_train: pyspark.sql.DataFrame

        Returns
        -------
        rdd

        """
        # Broadcast sampling parameters
        n_trees_bc, l_param_keys_bc = self.sc.broadcast(self.n_trees), self.sc.broadcast(self.param_keys)
        p_sampling_bc = self.sc.broadcast(self.compute_p_sampling(dfs_train.count()))

        # group sampled data
        rdd_group = dfs_train.rdd.flatMap(
            lambda row: [
                (k, {'Features': row['Features'], 'Label': row['Label']})
                for k in SparkRfSelector.sample_rows(n_trees_bc.value, p_sampling_bc.value, l_param_keys_bc.value)
            ]
        ) \
            .groupByKey()

        return rdd_group

    def predict_by_key(self, dfs_validation):
        """
        for each model (key), Predict the outcome of validation dataset

        Parameters
        ----------
        dfs_validation: pyspark.sql.DataFrame

        Returns
        -------
        rdd

        """
        l_models_bc = self.sc.broadcast(self.models)

        rdd_grouped = dfs_validation.rdd.flatMap(
            lambda row: [
                (m.key, {**m.predict_vector(row['Features'].array), "label": row['Label']})
                for m in l_models_bc.value
            ]
        ) \
            .groupByKey()

        return rdd_grouped

    def compute_p_sampling(self, n_sample):
        """
        Compute sampling rate used on dataset to feed one tree.

        The sampling rate depends on the size of the inputed dataframe versus the maximal size allowed to fit one tree
        and the maximum sampling rate allowed.

        Parameters
        ----------
        n_sample: int:
            Size of dataset used to fit the random forest

        Returns
        -------
        float:
            sampling rate floored at 2 decimal
        """
        p_sampling = min(self.max_p_sampling, float(self.max_sample_by_tree) / n_sample)

        # Floor p_sampling at 2 decimal
        return math.floor(p_sampling * 100) / 100.0

    @staticmethod
    def compute_param_keys(param_grid):
        """
        Compute combination of params as  a list of dict like string 'param1=value1,param2=value2,...'
        Parameters
        ----------
        param_grid: dict:
            Dict with param name as key and list of value to test as value.

        Returns
        -------
        list
            list of dict like string param cmbinations

        """
        it_param_keys = itertools.product(
            *[['='.join([k, str(v)]) for v in l_v] for k, l_v in param_grid.items()]
        )
        return [','.join(v) for v in it_param_keys]

    @staticmethod
    def sample_rows(n_tree, p_sampling, l_param_keys):
        """
        Augment parameter keys with random sampling of tree compounding Random forest.

        for each parameter key, for each tree compounding the random forest, add a corresponding tree_id to parameter
        with probability p_sampling.

        Parameters
        ----------
        n_tree: int
        p_sampling: float
        l_param_keys: list
            list of Dict like string of parameter of the random forest

        Returns
        -------
        list:
            list of keys
        """
        l_keys = []
        for pk in l_param_keys:
            l_tree_keys = [
                'tree_id={}'.format(i) for i in range(1, n_tree + 1) if bool(np.random.binomial(1, p_sampling))
            ]
            l_keys.extend([','.join([pk, tk]) for tk in l_tree_keys])

        return l_keys

    @staticmethod
    def process_data(k, l_features, l_labels, col_p_sample):
        """
        Prepare data, for decision tree training.

        Parameters
        ----------
        k: str:
            dict like string parameter that identifies combination of params of the random forest and the if of the
            decision tree.
        l_features: list:
            list of array containing the samples as array of features to use for training.
        l_labels
            list of label to use with training.
        col_p_sample: float:
            sampling proba to sample feature used for training.

        Returns
        -------
        tuple:
            data needed to fit the decision tree
        """
        # Get parameter tree as dict
        d_params = {x.split('=')[0]: x.split('=')[1] for x in k.split(',')}
        d_params_new = {
            k: v(d_params.get(k, None)) for k, v in SparkRfSelector.allowed_params.items() if k in d_params.keys()
        }

        # remove tree id from key
        new_key = ','.join(
            ['='.join([k, d_params.get(k, None)]) for k in SparkRfSelector.allowed_params.keys()
             if k in d_params.keys()]
        )

        # Build input and sample columns
        X, y = np.array(l_features), np.array(l_labels)
        ax_mask_features = np.random.binomial(1, col_p_sample, X.shape[1]).astype(bool)
        if ax_mask_features.sum() < SparkRfSelector.min_col_sampled:
            ax_inds = np.random.choice(
                np.arange(X.shape[1]), SparkRfSelector.min_col_sampled - ax_mask_features.sum(),
                replace=False
            )
            ax_mask_features[ax_inds] = True

        return X, y, new_key, d_params_new, ax_mask_features

    @staticmethod
    def fit_decision_tree(k, l_features, l_labels, col_p_sample):
        """
        Abstract method
        """
        pass

    @staticmethod
    def compute_score(key, it_values):
        """
        Abstract method
        """
        pass


class SparkRfClfSelector(SparkRfSelector):
    """

    """
    def fit(self, dfs_train):
        """
        Fit the random forest from a training dataset.

        Parameters
        ----------
        dfs_train: pyspark.sql.DataFrame

        Returns
        -------
        'SparkRfSelector'

        """
        # Distribute samples
        rdd_grouped = self.distribute_samples_for_train(dfs_train)

        # Broadcast model parameters
        col_p_sample_bc = self.sc.broadcast(self.col_p_sample)

        # Fit trees
        self.models = rdd_grouped.map(
            lambda x: SparkRfClfSelector.fit_decision_tree(
                x[0], [v['Features'].array for v in x[1]], [v['Label'] for v in x[1]], col_p_sample_bc.value
            )
        ) \
            .groupByKey()\
            .map(lambda x: RandomForestModel(x[0], list(x[1])))\
            .collect()

        return self

    def evaluate(self, dfs_validation):
        """
        Evaluate the Random forest with a validation dataset.

        Parameters
        ----------
        dfs_validation: pyspark.sql.DataFrame

        Returns
        -------
        'SparkRfSelector'

        """
        # Distribute samples
        rdd_predictions = self.predict_by_key(dfs_validation)

        l_scores = rdd_predictions \
            .map(lambda x: SparkRfClfSelector.compute_score(x[0], x[1])) \
            .collect()

        best_score_key = sorted(l_scores, key=lambda x: x[1], reverse=True)[0][0]
        self.best_model = [m for m in self.models if m.key == best_score_key][0]

        return self

    @staticmethod
    def fit_decision_tree(k, l_features, l_labels, col_p_sample):
        """
        Fit classifier decision tree.

        Parameters
        ----------
        k: str:
            dict like string parameter that identifies combination of params of the random forest and the if of the
            decision tree.
        l_features: list:
            list of array containing the samples as array of features to use for training.
        l_labels
            list of label to use with training.
        col_p_sample: float:
            sampling proba to sample feature used for training.

        Returns
        -------
        'DecisionTreeModel':

        """
        X, y, new_key, d_params, ax_mask_features = SparkRfClfSelector.process_data(
            k, l_features, l_labels, col_p_sample
        )

        clf = tree.DecisionTreeClassifier(**d_params) \
            .fit(X[:, ax_mask_features], y)

        return new_key, DecisionTreeModel(clf, ax_mask_features, mtype='clf')

    @staticmethod
    def compute_score(key, it_values):
        """

        Parameters
        ----------
        key
        it_values

        Returns
        -------

        """
        score = metrics.roc_auc_score(
            np.array([v['prediction'] for v in it_values]), np.array([v['label'] for v in it_values])
        )
        return key, score


class SparkRfRegSelector(SparkRfSelector):
    """"""

    def fit(self, dfs_train):
        """
        Fit the random forest from a training dataset.

        Parameters
        ----------
        dfs_train: pyspark.sql.DataFrame

        Returns
        -------
        'SparkRfSelector'

        """
        # Distribute samples
        rdd_grouped = self.distribute_samples_for_train(dfs_train)

        # Broadcast model parameters
        col_p_sample_bc = self.sc.broadcast(self.col_p_sample)

        # Fit trees
        self.models = rdd_grouped.map(
            lambda x: SparkRfRegSelector.fit_decision_tree(
                x[0], [v['Features'].array for v in x[1]], [v['Label'] for v in x[1]], col_p_sample_bc.value
            )
        ) \
            .groupByKey()\
            .map(lambda x: RandomForestModel(x[0], list(x[1])))\
            .collect()

        return self

    def evaluate(self, dfs_validation):
        """
        Evaluate the Random forest with a validation dataset.

        Parameters
        ----------
        dfs_validation: pyspark.sql.DataFrame

        Returns
        -------
        'SparkRfSelector'

        """
        # Distribute samples
        rdd_predictions = self.predict_by_key(dfs_validation)

        l_scores = rdd_predictions \
            .map(lambda x: SparkRfRegSelector.compute_score(x[0], x[1])) \
            .collect()

        best_score_key = sorted(l_scores, key=lambda x: x[1], reverse=True)[0][0]
        self.best_model = [m for m in self.models if m.key == best_score_key][0]

        return self

    @staticmethod
    def fit_decision_tree(k, l_features, l_labels, col_p_sample):
        """
        Fit regression tree.

        Parameters
        ----------
        k: str:
            dict like string parameter that identifies combination of params of the random forest and the if of the
            decision tree.
        l_features: list:
            list of array containing the samples as array of features to use for training.
        l_labels
            list of label to use with training.
        col_p_sample: float:
            sampling proba to sample feature used for training.

        Returns
        -------
        'DecisionTreeModel':

        """
        X, y, new_key, d_params, ax_mask_features = SparkRfClfSelector.prepare_data(
            k, l_features, l_labels, col_p_sample
        )

        clf = tree.DecisionTreeClassifier(**d_params) \
            .fit(X[:, ax_mask_features], y)

        return new_key, DecisionTreeModel(clf, ax_mask_features, mtype='reg')

    @staticmethod
    def compute_score(key, it_values):
        """

        Parameters
        ----------
        key
        it_values

        Returns
        -------

        """
        score = metrics.mean_squared_error(
            np.array([v['prediction'] for v in it_values]), np.array([v['label'] for v in it_values])
        )
        return key, score
