import pandas as pd

import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.decomposition import PCA
from openfe import OpenFE, transform
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder


import os
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import tree
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.optimize import curve_fit
from scipy.stats.stats import pearsonr
from sklearn.linear_model import Ridge
import matplotlib.patches as mpatches
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import mutual_info_classif

models = {'Logistic Regression': LogisticRegression(),
          'SVC': SVC(probability=True),
          'Random Forest' : RandomForestClassifier(),
          'AdaBoost' : AdaBoostClassifier() 
}

# Function to construct new features
def construct_features(df, operators):
    new_features = pd.DataFrame()
    columns = df.columns
    for col1 in columns:
        for col2 in columns:
            if col1 != col2:
                for op in operators:
                    new_col_name = f"{col1}_{op}_{col2}"
                    if op == '+':
                        new_features[new_col_name] = df[col1] + df[col2]
                    elif op == '-':
                        new_features[new_col_name] = df[col1] - df[col2]
                    elif op == '*':
                        new_features[new_col_name] = df[col1] * df[col2]
                    elif op == '/':
                        new_features[new_col_name] = df[col1] / (df[col2] + 1e-9)
    return new_features

def select_top_features(X, y, k=10):
    chi2_values, p_values = chi2(X, y)
    top_indices = np.argsort(chi2_values)[-k:]
    return X.iloc[:, top_indices]

def iterative_feature_construction(X, y, iterations=1, k=30):
    scaler = MinMaxScaler()
    operators = ['+', '-', '*', '/']
    X_combined = X.copy()
    for _ in range(iterations):
        new_features = construct_features(X_combined, operators)
        new_features = pd.DataFrame(scaler.fit_transform(new_features), columns=new_features.columns)
        top_features = select_top_features(new_features, y, k=k)
        X_combined = pd.concat([X_combined.reset_index(drop=True), top_features.reset_index(drop=True)], axis=1)
    return X_combined

class IFCModel:
    def __init__(self, data, target, numeric_features, categorical_features):
        self.data = data
        self.target = target
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

    def preprocess_data(self):
        X = self.data.drop([self.target], axis=1)
        y = self.data[self.target].values.ravel()

        categorical_transformer = OrdinalEncoder()
        X_encoded = categorical_transformer.fit_transform(X[self.categorical_features])
        X[self.categorical_features] = X_encoded
        
        X_iterative = iterative_feature_construction(X, y, iterations=2)
        
        X_train, X_test, y_train, y_test = train_test_split(X_iterative, y, test_size=0.2, random_state=42, stratify=y)

        return X_train, X_test, y_train, y_test


    def fit_predict_evaluate(self):
        X_train, X_test, y_train, y_test = self.preprocess_data()

        results_accuracy = {}
        results_roc_auc = {}

        for name, model in models.items():
          model.fit(X_train, y_train)
          y_pred = model.predict(X_test)

          accuracy = accuracy_score(y_test, y_pred)
          roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

          results_accuracy[name] = accuracy
          results_roc_auc[name] = roc_auc

        results_accuracy_series = pd.Series(results_accuracy)
        results_roc_auc_series = pd.Series(results_roc_auc)

        return results_accuracy_series, results_roc_auc_series