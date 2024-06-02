import os
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import tree
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
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

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
import os

models = {'Logistic Regression': LogisticRegression(),
          'SVC': SVC(probability=True),
          'Random Forest' : RandomForestClassifier(),
          'AdaBoost' : AdaBoostClassifier() 
}

def is_csv_empty(filename):
    # Check if the CSV file is empty
    if not os.path.exists(filename) or os.stat(filename).st_size == 0:
        return True
    return False

def linear(TR, TST):
    a, b = TR.shape
    c, d = TST.shape

    if is_csv_empty('linear_correlated.csv'):
        return

    dataset = pd.read_csv('linear_correlated.csv', header=None)
    val = dataset.values
    aa, _ = val.shape

    predicted_train = np.zeros((a, 0), dtype=float)
    predicted_test = np.zeros((c, 0), dtype=float)
    predicted_train_error = np.zeros((a, 0), dtype=float)
    predicted_test_error = np.zeros((c, 0), dtype=float)

    ridge = Ridge(alpha=1.0)

    scaler = StandardScaler()
    TR_scaled = scaler.fit_transform(TR)
    TST_scaled = scaler.transform(TST)

    for j in range(aa):
        rr = TR_scaled[:, int(val[j][0])].reshape(-1, 1)
        ss = TR_scaled[:, int(val[j][1])]
        tt = TST_scaled[:, int(val[j][0])].reshape(-1, 1)
        uu = TST_scaled[:, int(val[j][1])]

        y_train = cross_val_predict(ridge, rr, ss, cv=5).reshape(-1, 1)
        ridge.fit(rr, ss)
        y_test = ridge.predict(tt).reshape(-1, 1)

        predicted_train = np.hstack([predicted_train, y_train])
        predicted_test = np.hstack([predicted_test, y_test])

        diff_train = (ss.reshape(-1, 1) - y_train)
        diff_test = (uu.reshape(-1, 1) - y_test)
        predicted_train_error = np.hstack([predicted_train_error, diff_train])
        predicted_test_error = np.hstack([predicted_test_error, diff_test])

    predicted_train_final = np.hstack([predicted_train, predicted_train_error])
    predicted_test_final = np.hstack([predicted_test, predicted_test_error])

    save_to_csv("related_lineartest.csv", predicted_test_final)
    save_to_csv("related_lineartrain.csv", predicted_train_final)

def save_to_csv(filename, data):
    if os.path.exists(filename):
        os.remove(filename)
    np.savetxt(filename, data, delimiter=",", fmt="%s")


from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os

def stable(ress, test, labels, lasso_threshold=0.1, top_k_features=10):
    x, y = ress.shape
    names = np.arange(y)

    rlasso = Lasso(alpha=0.1)
    rlasso.fit(ress, labels)

    val = sorted(zip(map(lambda x: round(x, 4), rlasso.coef_), names), key=lambda x: x[0], reverse=True)
    print("len of val:", len(val))
    global nc_val
    nc_val += len(val)

    finale = [s for r, s in val if r > lasso_threshold]
    if not finale:  # If no features are selected, choose at least one
        finale.append(names[0])

    print("Total features after stability selection:", len(finale))
    global stable_val
    stable_val += len(finale)

    dataset1 = ress[:, finale]
    dataset3 = test[:, finale]

    # Output file handling
    if os.path.exists("stable_testfeatures.csv"):
        os.remove("stable_testfeatures.csv")
    if os.path.exists("stable_trainfeatures.csv"):
        os.remove("stable_trainfeatures.csv")

    np.savetxt("stable_testfeatures.csv", dataset3, delimiter=",", fmt="%s")
    np.savetxt("stable_trainfeatures.csv", dataset1, delimiter=",", fmt="%s")

    forest = RandomForestClassifier(n_estimators=50, random_state=42)
    forest.fit(ress, labels)
    importances = forest.feature_importances_

    indices = np.argsort(importances)[-top_k_features:]

    ress_top_k = ress[:, indices]
    test_top_k = test[:, indices]

    if os.path.exists("ensemble_testfeatures.csv"):
        os.remove("ensemble_testfeatures.csv")
    if os.path.exists("ensemble_trainfeatures.csv"):
        os.remove("ensemble_trainfeatures.csv")

    np.savetxt("ensemble_testfeatures.csv", test_top_k, delimiter=",", fmt="%s")
    np.savetxt("ensemble_trainfeatures.csv", ress_top_k, delimiter=",", fmt="%s")

    print("Total features after ensemble selection:", top_k_features)
    global ensemble_val
    ensemble_val += top_k_features

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import os

def original_rf_feature_selection(ress, test, labels):
    global len_orig_rf
    x, y = ress.shape

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(ress, labels)

    selector = SelectFromModel(rf, threshold='median', prefit=True)
    selected_features = selector.get_support(indices=True)

    print("Selected features after Random Forest:")
    len_orig_rf += len(selected_features)
    print(len(selected_features))

    dataset1 = ress[:, selected_features]
    dataset3 = test[:, selected_features]

    train_file = "rf_selected_trainfeatures.csv"
    test_file = "rf_selected_testfeatures.csv"

    if os.path.exists(train_file):
        os.remove(train_file)
    if os.path.exists(test_file):
        os.remove(test_file)

    np.savetxt(train_file, dataset1, delimiter=",", fmt="%s")
    np.savetxt(test_file, dataset3, delimiter=",", fmt="%s")

def rank(X1,y):
    forest = ExtraTreesClassifier(n_estimators=250,random_state=0)
    forest.fit(X1, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
import os

def nonlinear(TR, TST):
    dataset = pd.read_csv('nonlinear_correlated.csv', header=None)
    val = dataset.values
    a, b = TR.shape
    c, d = TST.shape
    aa, _ = val.shape

    predicted_train = np.zeros((a, 0), dtype=float)
    predicted_test = np.zeros((c, 0), dtype=float)
    predicted_train_error = np.zeros((a, 0), dtype=float)
    predicted_test_error = np.zeros((c, 0), dtype=float)

    gbr = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, max_depth=5, random_state=42)

    scaler = StandardScaler()
    TR_scaled = scaler.fit_transform(TR)
    TST_scaled = scaler.transform(TST)

    for j in range(aa):
        rr = TR_scaled[:, int(val[j][0])].reshape(-1, 1)
        ss = TR_scaled[:, int(val[j][1])]
        tt = TST_scaled[:, int(val[j][0])].reshape(-1, 1)
        uu = TST_scaled[:, int(val[j][1])]

        y_train = cross_val_predict(gbr, rr, ss, cv=5).reshape(-1, 1)
        gbr.fit(rr, ss)
        y_test = gbr.predict(tt).reshape(-1, 1)

        predicted_train = np.hstack([predicted_train, y_train])
        predicted_test = np.hstack([predicted_test, y_test])

        diff_train = (ss.reshape(-1, 1) - y_train)
        diff_test = (uu.reshape(-1, 1) - y_test)
        predicted_train_error = np.hstack([predicted_train_error, diff_train])
        predicted_test_error = np.hstack([predicted_test_error, diff_test])

    predicted_train_final = np.hstack([predicted_train, predicted_train_error])
    predicted_test_final = np.hstack([predicted_test, predicted_test_error])

    save_to_csv("related_nonlineartrain.csv", predicted_train_final)
    save_to_csv("related_nonlineartest.csv", predicted_test_final)

def save_to_csv(filename, data):
    if os.path.exists(filename):
        os.remove(filename)
    np.savetxt(filename, data, delimiter=",")

def distcorr(X, Y):
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor

def dependent(x, th1):
    linear_correlated = []
    nonlinear_correlated = []
    m, n = x.shape
    linear_count = 0
    nonlinear_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            pearson_corr, _ = pearsonr(x[:, i], x[:, j])
            distance_corr = distcorr(x[:, i], x[:, j])

            if distance_corr >= th1:
                linear_correlated.append((i, j))
                linear_count += 1
            elif 0 < distance_corr < 0.7:
                nonlinear_correlated.append((i, j))
                nonlinear_count += 1

    save_correlated_features('linear_correlated.csv', linear_correlated)
    save_correlated_features('nonlinear_correlated.csv', nonlinear_correlated)

    print(f"Number of linear correlated features: {linear_count}")
    print(f"Number of nonlinear correlated features: {nonlinear_count}")

    return linear_count

def save_correlated_features(filename, data):
    if os.path.exists(filename):
        os.remove(filename)
    np.savetxt(filename, data, delimiter=",", fmt="%d")


class ALModModel:
    def __init__(self, data, target, numeric_features, categorical_features):
        self.data = data
        self.target = target
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

    def preprocess_data(self):
        X = self.data.drop([self.target], axis=1)
        y = self.data[self.target]

        categorical_transformer = OrdinalEncoder()

        X_train_encoded = categorical_transformer.fit_transform(X[self.categorical_features])
        X[categorical_transformer.get_feature_names_out()] = X_train_encoded
    
        if self.data.shape[0] < 1000:
             X_train, X_test, y_train, y_test = train_test_split(X, y,
                                    train_size=600, test_size=168, random_state=42, stratify=y)
        elif self.data.shape[0] == 1000:
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=800, test_size=200, random_state=42, stratify=y)
        else:    
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=2000, test_size=500, random_state=42, stratify=y)
            
        train = X_train.copy().values
        test = X_test.copy().values
        trainY = y_train.copy().values
        testY = y_test.copy().values

        return train, test, trainY, testY


    def fit_predict_evaluate(self):
        train, test, trainY, testY = self.preprocess_data()

        global len_orig_rf, nc_val, stable_val, ensemble_val
        len_orig_rf=0
        nc_val=0
        stable_val=0
        ensemble_val=0
        
        original_rf_feature_selection(train,test,trainY)  # No normalization needed for original training & testing
        original_ig_train=pd.read_csv('rf_selected_trainfeatures.csv',header=None)
        original_ig_test=pd.read_csv('rf_selected_testfeatures.csv',header=None)

        original_ig_train=original_ig_train.values
        original_ig_test=original_ig_test.values

        linear_count = dependent(original_ig_train, 0.8)

        nonlinear(original_ig_train,original_ig_test)
        
        a3=pd.read_csv('related_nonlineartest.csv',header=None)
        a4=pd.read_csv('related_nonlineartrain.csv',header=None)

        if linear_count != 0:
            linear(original_ig_train,original_ig_test)
            a1=pd.read_csv('related_lineartest.csv',header=None)
            a2=pd.read_csv('related_lineartrain.csv',header=None)
            r4=np.hstack([a2,a4])
            r3=np.hstack([a1,a3])
        else:
            r4=a4
            r3=a3
        
        scaler=StandardScaler().fit(r4)
        p2=scaler.transform(r4)
        p1=scaler.transform(r3)

        stable(p2,p1,trainY)
        f1=pd.read_csv('ensemble_trainfeatures.csv',header=None)
        f2=pd.read_csv('ensemble_testfeatures.csv',header=None)

        scaler=StandardScaler().fit(f1)
        e_f1=scaler.transform(f1)
        e_f2=scaler.transform(f2)

        x1X=np.hstack([test,f2])
        x2X=np.hstack([train,f1])

        scaler=StandardScaler().fit(x2X)
        x2=scaler.transform(x2X)
        x1=scaler.transform(x1X)

        y1Y=np.hstack([test, f2])
        y2Y=np.hstack([train, f1])

        scaler=StandardScaler().fit(y2Y)  # Again normalization of the complete combined feature pool
        y2=scaler.transform(y2Y)          # note - when features need to be merged with R2R, we need to do normalization.
        y1=scaler.transform(y1Y)

        st_f1=pd.read_csv('stable_trainfeatures.csv',header=None)
        st_f2=pd.read_csv('stable_testfeatures.csv',header=None)

        st_x1X=np.hstack([original_ig_test, st_f2])  # original test features, selected by IG, f2 is feature space after stability selection.
        st_x2X=np.hstack([original_ig_train, st_f1])

        scaler=StandardScaler().fit(st_x2X)  # Again normalization of the complete combined feature pool
        st_x2=scaler.transform(st_x2X)          # note - when features need to be merged with R2R, we need to do normalization.
        st_x1=scaler.transform(st_x1X)

        best_roc_auc = 0
        best_accuracy = 0
        best_model = ''

        for name, model in models.items():
            model.fit(original_ig_train,trainY)
            y_out= model.predict(original_ig_test)
            roc = roc_auc_score(testY, model.predict_proba(original_ig_test)[:,1])
            accuracy = accuracy_score(testY, y_out)
            if best_roc_auc < roc:
                best_roc_auc = roc
                best_accuracy = accuracy
                best_model = name + ' Results on (Original + IG) Stable Features'


        for name, model in models.items():
            model.fit(p2,trainY)
            y_out= model.predict(p1)
            roc = roc_auc_score(testY, model.predict_proba(p1)[:,1])
            accuracy = accuracy_score(testY, y_out)
            if best_roc_auc < roc:
                best_roc_auc = roc
                best_accuracy = accuracy
                best_model = name + ' Results on just the Newly constructed Features'


        for name, model in models.items():
            model.fit(e_f1,trainY)
            y_out= model.predict(e_f2)
            roc = roc_auc_score(testY, model.predict_proba(e_f2)[:,1])
            accuracy = accuracy_score(testY, y_out)
            if best_roc_auc < roc:
                best_roc_auc = roc
                best_accuracy = accuracy
                best_model = name + ' Results after Newly constructed Features with feature selection'

        for name, model in models.items():
            model.fit(y2,trainY)
            output= model.predict(y1)
            roc = roc_auc_score(testY, model.predict_proba(y1)[:,1])
            accuracy = accuracy_score(testY, y_out)
            if best_roc_auc < roc:
                best_roc_auc = roc
                best_accuracy = accuracy
                best_model = name + ' Results after Newly constructed Features + Original with feature selection'
            

        for name, model in models.items():
            model.fit(x2,trainY)
            y_out= model.predict(x1)
            roc = roc_auc_score(testY, model.predict_proba(x1)[:,1])
            accuracy = accuracy_score(testY, y_out)
            if best_roc_auc < roc:
                best_roc_auc = roc
                best_accuracy = accuracy
                best_model = name + ' Results when full architecture of AutoLearn is followed (IG & then Stability in feature selection)'

        for name, model in models.items():
            model.fit(st_x2,trainY)
            y_out= model.predict(st_x1)
            roc = roc_auc_score(testY, model.predict_proba(st_x1)[:,1])
            accuracy = accuracy_score(testY, y_out)
            if best_roc_auc < roc:
                best_roc_auc = roc
                best_accuracy = accuracy
                best_model = name + ' Results when full architecture of AutoLearn is followed (Stability only in feature selection)'
            
        return best_model, best_accuracy, best_roc_auc
