# Global import
import copy
import numpy as np
from pyspark.sql.types import StructField, FloatType, ArrayType


class DecisionTreeModel:
    """
    Simple Decision tree.

    Wrap the sk learn decision tree.

    """
    def __init__(self, sk_dt, ax_mask_features, mtype="clf"):
        """
        Constructor.

        Parameters
        ----------
        sk_dt: sklear.tree:
        ax_mask_features: np.array
            mask feature for this tree.
        mtype: str:
            type of model. in ['clf', 'reg']

        """
        self.sk_dt = sk_dt
        self.mask_features = ax_mask_features
        self.mtype = mtype

    def predict(self, x):
        """
        Predict a single array of feature.

        Parameters
        ----------
        x: np.array

        Returns
        -------
        list:
            probability of each class.
        """
        x_masked = x[np.newaxis, self.mask_features]

        if self.mtype == 'clf':
            return list(self.sk_dt.predict_proba(x_masked)[0])

        elif self.mtype == 'reg':
            return self.sk_dt.predict(x_masked)[0]

        else:
            raise ValueError('Wrong mtype {}'.format(self.mtype))


class RandomForestModel:

    def __init__(self, key, l_decision_trees, mtype="clf"):
        self.decision_tres = list(l_decision_trees)
        self.key = key
        self.mtype = mtype

    @staticmethod
    def from_path(hdfs_path):
        """
        Instantiate the model from hdfs location.

        Parameters
        ----------
        hdfs_path

        Returns
        -------
        'RandomForestModel'
        """
        pass

    def predict_vector(self, x):
        """
        Predict a single array of feature.

        Parameters
        ----------
        x: Array like:

        Returns
        -------
        dict:
        """
        if self.mtype == 'clf':
            ax_prediction = np.array([dt.predict(x) for dt in self.decision_tres]).mean(axis=0)
            return {"prediction": float(np.argmax(ax_prediction)), "probability": [float(x) for x in ax_prediction]}

        elif self.mtype == 'reg':
            pred = np.array([dt.predict(x) for dt in self.decision_tres]).mean()
            return {"prediction": float(pred)}

        else:
            raise ValueError('Wrong mtype {}'.format(self.mtype))

    def predict(self, dfs_data):
        """

        Parameters
        ----------
        dfs_data

        Returns
        -------

        """
        # Create new schema
        new_schema = self.get_new_schema(dfs_data.schema)

        # Predict
        dfs_data = dfs_data.rdd \
            .map(lambda row: {**row.asDict(), **self.predict_vector(row['Features'])})\
            .toDF(new_schema)

        return dfs_data

    def get_new_schema(self, schema):
        """

        Parameters
        ----------
        schema

        Returns
        -------

        """
        new_schema = copy.deepcopy(schema)
        new_schema.add(
            StructField('prediction', FloatType(), nullable=True)
        )

        if self.mtype == 'clf':
            new_schema.add(
                StructField("probability", ArrayType(FloatType()), True)
            )

        return new_schema

    def save_model(self, hdfs_path):
        """
        Save model to Hdfs.

        Parameters
        ----------
        hdfs_path

        Returns
        -------

        """
        pass

