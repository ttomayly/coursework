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

import csv

def is_csv_empty(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        try:
            # Attempt to read the first row
            first_row = next(reader)
            # If there is a first row, return False (not empty)
            return False
        except StopIteration:
            # If there is no first row, return True (empty)
            return True
clf = Ridge(alpha=1.0)
cnt=0
ans=[]

np.random.seed(7)  # to ensure that everytime results are same

###########################################################################
                  # Function to randomly shuffle the data
###########################################################################


def shuffle(df, n=1, axis=0):
    df = df.copy()
    for _ in range(n):
      df.apply(np.random.shuffle, axis=axis)
    return df


###########################################################################
                   # Function to define quadratic curve
###########################################################################


def curve(x,a,b,c):
    return a*(x**2) + b*x + c


###########################################################################
              # Function to compute the distance correlation
###########################################################################


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

def linear(TR,TST):
    a,b=TR.shape
    c,d=TST.shape
    if is_csv_empty('linear_correlated.csv'):
      return
    dataset=pd.read_csv('linear_correlated.csv',header=None)
    val=dataset.values
    aa,bb=val.shape

    # Feature matrix initialized that stores features constructed
    predicted_train=np.zeros((a,len(ans)),dtype=float)
    predicted_test=np.zeros((c,len(ans)),dtype=float)
    predicted_train_final=np.zeros(2*(a,len(ans)),dtype=float)
    predicted_test_final=np.zeros(2*(c,len(ans)),dtype=float)
    predicted_train_error=np.zeros((a,len(ans)),dtype=float)
    predicted_test_error=np.zeros((c,len(ans)),dtype=float)

    for j in range(0,aa):
           rr,ss=np.array(TR[:,(int)(val[j][0])][:,np.newaxis]),np.array(TR[:,(int)(val[j][1])])
           tt,uu=np.array(TST[:,(int)(val[j][0])][:,np.newaxis]),np.array(TST[:,(int)(val[j][1])])
           y_train=clf.fit(rr,ss).predict(rr)[:,np.newaxis]
           y_test=clf.fit(rr,ss).predict(tt)[:,np.newaxis]
           predicted_train=np.hstack([predicted_train,y_train])
           predicted_test=np.hstack([predicted_test,y_test])

           dd=ss[:,np.newaxis]
           ee=uu[:,np.newaxis]
           diff_train=(dd-y_train)
           diff_test=(ee-y_test)
           predicted_train_error=np.hstack([predicted_train_error,diff_train])
           predicted_test_error=np.hstack([predicted_test_error,diff_test])
           #predicted_test=np.hstack([predicted_test,clf.coef_*TST[:,val[j][1]][:, np.newaxis]+clf.intercept_])
           #predicted_train=np.hstack([predicted_train,clf.coef_*TR[:,val[j][1]][:, np.newaxis]+clf.intercept_])
    predicted_train_final=np.hstack([predicted_train,predicted_train_error])
    predicted_test_final=np.hstack([predicted_test,predicted_test_error])
    # Saving constructed features finally to a file

    if os.path.exists("related_lineartest.csv"):                          # Name of Ouput file generated
       os.remove("related_lineartest.csv")

    if os.path.exists('related_lineartrain.csv'):                          # Name of Ouput file generated
       os.remove('related_lineartrain.csv')

    with open("related_lineartest.csv", "wb") as myfile:
            np.savetxt(myfile,predicted_test_final,delimiter=",",fmt="%s")
    with open("related_lineartrain.csv", "wb") as myfile:
            np.savetxt(myfile,predicted_train_final,delimiter=",",fmt="%s")

def dependent(x,th1):
    ans=[]
    ans1=[]
    m,n=x.shape
    cnt=0
    cnt1=0
    for i in range(0,n):
       for j in range(0,n):
           if (i!=j):
              a,b=pearsonr(x[:,i],x[:,j])
              if(distcorr(np.array(x[:,i]),np.array(x[:,j]))>=th1):
               a1=i,j
               ans.append(a1)
               cnt=cnt+1
              elif(distcorr(np.array(x[:,i]),np.array(x[:,j]))>0 and distcorr(np.array(x[:,i]),np.array(x[:,j]))<0.7):
               zz=i,j
               ans1.append(zz)
               cnt1=cnt1+1

       #print(i)
    if os.path.exists('linear_correlated.csv'):                             # Name of Ouput file generated
       os.remove('linear_correlated.csv')
    if os.path.exists('nonlinear_correlated_{}.csv'):                          # Name of Ouput file generated
       os.remove('nonlinear_correlated_{}.csv')

    np.savetxt("linear_correlated.csv",ans,delimiter=",",fmt="%.5f")
    np.savetxt("nonlinear_correlated.csv",ans1,delimiter=",",fmt="%s")


def nonlinear(TR,TST):
    a,b=TR.shape
    c,d=TST.shape
    dataset=pd.read_csv('nonlinear_correlated.csv',header=None)
    val=dataset.values
    aa,bb=val.shape

    # Feature matrix initialized that stores features constructed
    predicted_train=np.zeros((a,len(ans)),dtype=float)
    predicted_test=np.zeros((c,len(ans)),dtype=float)
    predicted_train_final=np.zeros(2*(a,len(ans)),dtype=float)
    predicted_test_final=np.zeros(2*(c,len(ans)),dtype=float)

    predicted_train_error=np.zeros((a,len(ans)),dtype=float)
    predicted_test_error=np.zeros((c,len(ans)),dtype=float)

    svr_rbf = KernelRidge(alpha=1.0, coef0=1, degree=3, gamma=None, kernel='rbf',
                          kernel_params=None)

    #svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    for j in range(0,aa):
        rr,ss=np.array(TR[:,(int)(val[j][0])][:,np.newaxis]),np.array(TR[:,(int)(val[j][1])])
        tt,uu=np.array(TST[:,(int)(val[j][0])][:,np.newaxis]),np.array(TST[:,(int)(val[j][1])])

        y_train=svr_rbf.fit(rr,ss).predict(rr)[:,np.newaxis]
        y_test=svr_rbf.fit(rr,ss).predict(tt)[:,np.newaxis]
        predicted_train=np.hstack([predicted_train,y_train])
        predicted_test=np.hstack([predicted_test,y_test])


        dd=ss[:,np.newaxis]
        ee=uu[:,np.newaxis]
        diff_train=(dd-y_train)
        diff_test=(ee-y_test)

        predicted_train_error=np.hstack([predicted_train_error,diff_train])
        predicted_test_error=np.hstack([predicted_test_error,diff_test])
        '''
        popt, pcov = curve_fit(curve,rr,ss)
        predicted_test=np.hstack([predicted_test,float(popt[0])*(TST[:,val[j][1]][:, np.newaxis]**2)+float(popt[1])*TST[:,val[j][1]][:,    np.newaxis]+float(popt[2])])
        predicted_train=np.hstack([predicted_train,float(popt[0])*(TR[:,val[j][1]][:, np.newaxis]**2)+float(popt[1])*TR[:,val[j][1]][:, np.newaxis]+float(popt[2])])
        '''
    predicted_train_final=np.hstack([predicted_train,predicted_train_error])
    predicted_test_final=np.hstack([predicted_test,predicted_test_error])

    if os.path.exists("related_nonlineartest.csv"):                          # Name of Ouput file generated
       os.remove("related_nonlineartest.csv")

    if os.path.exists('related_nonlineartrain.csv'):                          # Name of Ouput file generated
       os.remove('related_nonlineartrain.csv')

    # Saving constructed features finally to a file
    with open("related_nonlineartest.csv", "wb") as myfile:
            np.savetxt(myfile,predicted_test_final,delimiter=",")
    with open("related_nonlineartrain.csv", "wb") as myfile:
            np.savetxt(myfile,predicted_train_final,delimiter=",")

#############################################################################
    # Function to select features from the original ones using I.G
#############################################################################

def original_ig(ress,test,labels):   # ress is training data
    x,y = ress.shape
    names = np.arange(y)

    ress_new = SelectKBest(mutual_info_classif, k='all')
    ress_new.fit_transform(ress, labels)

    original_features=sorted(zip(map(lambda x: round(x, 4), ress_new.scores_),
    			 names), reverse=True)

    finale=[]
    for i in range(0,len(original_features)):
        r,s=original_features[i]
        if(r>0):   # This is eta-o
          finale.append(s)

        #finale.append(s)


    global len_orig_ig
    len_orig_ig += len(finale)

    dataset1=np.zeros((len(ress),len(finale)),dtype=float)
    dataset3=np.zeros((len(test),len(finale)),dtype=float)
    dataset1=ress[:,finale]
    dataset3=test[:,finale]
    #dataset3=test.iloc[:,finale]

    if os.path.exists("original_ig_testfeatures.csv"):                           # Name of Ouput file generated
       os.remove("original_ig_testfeatures.csv")
    if os.path.exists("original_ig_trainfeatures.csv"):                          # Name of Ouput file generated
       os.remove("original_ig_trainfeatures.csv")

    with open("original_ig_testfeatures.csv", "wb") as myfile:
            np.savetxt(myfile,dataset3,delimiter=",",fmt="%s")
    with open("original_ig_trainfeatures.csv", "wb") as myfile:
            np.savetxt(myfile,dataset1,delimiter=",",fmt="%s")

from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn import linear_model
def stable(ress, test, labels):
    x, y = ress.shape
    names = np.arange(y)
    rlasso = linear_model.Lasso()
    rlasso.fit(ress, labels)

    # Stability Selection
    val = sorted(zip(map(lambda x: round(x, 4), rlasso.coef_), names), key=lambda x: x[0], reverse=True)
    global nc_val
    nc_val += len(val)

    finale = [s for r, s in val if r > 0.1]  # Select features with scores > 0.1
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

    # Ensemble Selection
    ress_new = SelectKBest(mutual_info_classif, k='all')
    ress_new.fit_transform(ress[:, finale], labels)

    feats = sorted(zip(map(lambda x: round(x, 4), ress_new.scores_), names), key=lambda x: x[0], reverse=True)
    print("Total features after 2nd phase selection:", len(feats))
    global ensemble_val
    ensemble_val += len(feats)

    ensemble_finale = [s for r, s in feats if r > 0]  # Select features with scores > 0 (eta-o)
    if not ensemble_finale:  # If no features are selected, choose at least one
        ensemble_finale.append(names[0])

    dataset2 = ress[:, ensemble_finale]
    dataset4 = test[:, ensemble_finale]

    if os.path.exists("ensemble_testfeatures.csv"):
        os.remove("ensemble_testfeatures.csv")
    if os.path.exists("ensemble_trainfeatures.csv"):
        os.remove("ensemble_trainfeatures.csv")

    np.savetxt("ensemble_testfeatures.csv", dataset4, delimiter=",", fmt="%s")
    np.savetxt("ensemble_trainfeatures.csv", dataset2, delimiter=",", fmt="%s")

models = {'Logistic Regression': LogisticRegression(),
          'SVC': SVC(probability=True),
          'Random Forest' : RandomForestClassifier(),
          'AdaBoost' : AdaBoostClassifier() 
}

class ALModel:
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
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=800, test_size=200, random_state=42, stratify=y)
            
        print(X_train, X_test, y_train, y_test)
        train = X_train.copy().values
        test = X_test.copy().values
        trainY = y_train.copy().values
        testY = y_test.copy().values

        return train, test, trainY, testY


    def fit_predict_evaluate(self):
        train, test, trainY, testY = self.preprocess_data()

        global len_orig_ig, nc_val, stable_val, ensemble_val
        len_orig_ig=0
        nc_val=0
        stable_val=0
        ensemble_val=0
        
        print(train,test,trainY)
        original_ig(train,test,trainY)  # No normalization needed for original training & testing
        original_ig_train=pd.read_csv('original_ig_trainfeatures.csv',header=None)
        original_ig_test=pd.read_csv('original_ig_testfeatures.csv',header=None)

        original_ig_train=original_ig_train.values
        original_ig_test=original_ig_test.values

        dependent(original_ig_train, 0.5)
        linear(original_ig_train,original_ig_test)
        nonlinear(original_ig_train,original_ig_test)

        a1=pd.read_csv('related_lineartest.csv',header=None)
        a2=pd.read_csv('related_lineartrain.csv',header=None)
        a3=pd.read_csv('related_nonlineartest.csv',header=None)
        a4=pd.read_csv('related_nonlineartrain.csv',header=None)

        r4=np.hstack([a2,a4])
        r3=np.hstack([a1,a3])
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

        results_accuracy = {}
        results_roc_auc = {}

        for name, model in models.items():
          model.fit(st_x2, trainY)
          y_pred = model.predict(st_x1)

          accuracy = accuracy_score(testY, y_pred)
          roc_auc = roc_auc_score(testY, model.predict_proba(st_x1)[:, 1])

          results_accuracy[name] = accuracy
          results_roc_auc[name] = roc_auc

        results_accuracy_series = pd.Series(results_accuracy)
        results_roc_auc_series = pd.Series(results_roc_auc)

        return results_accuracy_series, results_roc_auc_series