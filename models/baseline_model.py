from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, roc_auc_score


import numpy as np
import pandas as pd

from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import linear_model


from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.feature_selection import mutual_info_classif

models = {'Logistic Regression': LogisticRegression(),
          'SVC': SVC(probability=True),
          'Random Forest' : RandomForestClassifier(),
          'AdaBoost' : AdaBoostClassifier() 
}

class BaseModel:
    def __init__(self, data, target, numeric_features, categorical_features):
        self.data = data
        self.target = target
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

    def preprocess_data(self):
        X = self.data.drop([self.target], axis=1)
        y = self.data[self.target]

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            random_state=42)

        numeric_transformer = StandardScaler()
        categorical_transformer = OrdinalEncoder()

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])

        return X_train, X_test, y_train, y_test, preprocessor

    def fit_predict_evaluate(self):
        X_train, X_test, y_train, y_test, preprocessor = self.preprocess_data()

        results_accuracy = {}
        results_roc_auc = {}

        for name, model in models.items():
          pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)])
          pipeline.fit(X_train, y_train)
          y_pred = pipeline.predict(X_test)

          accuracy = accuracy_score(y_test, y_pred)
          roc_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])

          results_accuracy[name] = accuracy
          results_roc_auc[name] = roc_auc

        results_accuracy_series = pd.Series(results_accuracy)
        results_roc_auc_series = pd.Series(results_roc_auc)

        return results_accuracy_series, results_roc_auc_series