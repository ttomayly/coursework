from models.baseline_model import BaseModel
from models.autofeat_model import AutoFeatModel
from models.openfe_model import OpenFEModel
from models.ifc_model import IFCModel
from models.al_model import ALModel
from models.al_mod_model import ALModModel
from models.tpot_model import TPOTModel
from models.safe_model import SAFEModel

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


from data_transformer import DataTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import numpy as np

data_wage = DataTransformer('datasets/wage.csv', 'class')
data_wage.create_binary_class('class', '>50K')
data_wage.drop_columns(['id', 'education', 'fnlwgt'])

data_marketing = DataTransformer('datasets/bank-marketing.csv', 'Class')
data_marketing.drop_columns(['id'])
data_marketing.change_target(2)

data_promotion = DataTransformer('datasets/promotion.csv', 'is_promoted')
data_promotion.drop_columns(['employee_id', 'education', 'previous_year_rating'])

data_eye = DataTransformer('datasets/eye.csv', 'eyeDetection')

data_diabetes = DataTransformer('datasets/diabetes.csv', 'class')
data_diabetes.drop_columns(['id'])

data_credit = DataTransformer('datasets/credit-g.csv', 'class')
data_credit.drop_columns(['id'])

datasets = {
    # 'bank-marketing': data_marketing,
    # 'wage': data_wage,
    # 'promotion': data_promotion,
    'diabetes': data_diabetes,
    'eeg_eye': data_eye,
    # 'credit': data_credit
}


def dataset_summary(datasets):
    summary_data = {
        'Dataset': [],
        'Rows': [],
        'Features': [],
        'Categorical': [],
        'Numeric': [],
        '% of 1': []
    }

    for name, df in datasets.items():
        summary_data['Dataset'].append(name)
        summary_data['Rows'].append(df.data.shape[0])
        summary_data['Features'].append(df.data.shape[1])
        summary_data['Categorical'].append(sum(df.data.dtypes == 'object'))
        summary_data['Numeric'].append(sum(np.issubdtype(dtype, np.number) for dtype in df.data.dtypes))
        summary_data['% of 1'].append(df.get_target_distribution())

    summary_df = pd.DataFrame(summary_data)
    return summary_df

# summary = dataset_summary(datasets)

# print(summary)

models = {'Logistic Regression': LogisticRegression(),
          'SVC': SVC(probability=True),
          'Random Forest' : RandomForestClassifier(),
          'AdaBoost' : AdaBoostClassifier() 
}

results = {
    'bank-marketing': [],
    'wage': [],
    'promotion': [],
    'diabetes': [],
    'eeg_eye': [],
    'credit': []
}

result_auc_base = pd.DataFrame(index=models.keys())
result_ac_base = pd.DataFrame(index=models.keys())

result_auc_openfe = pd.DataFrame(index=models.keys())
result_ac_openfe = pd.DataFrame(index=models.keys())

result_auc_ifc = pd.DataFrame(index=models.keys())
result_ac_ifc = pd.DataFrame(index=models.keys())

result_auc_al = pd.DataFrame(index=models.keys())
result_ac_al = pd.DataFrame(index=models.keys())

results_accuracy_safe = {}
results_roc_auc_safe = {}

results_accuracy_af = {}
results_roc_auc_af = {}

results_accuracy_tpot = {}
results_roc_auc_tpot = {}


for dataset_name, data in datasets.items():
    base_model = BaseModel(data=data.data,
                                target=data.target,
                                numeric_features=data.numeric_features,
                                categorical_features=data.categorical_features)

    results_accuracy_series_base, results_roc_auc_series_base = base_model.fit_predict_evaluate()

    results_accuracy_series_base.name = dataset_name
    results_roc_auc_series_base.name = dataset_name

    result_ac_base = pd.concat([result_ac_base, results_accuracy_series_base], axis=1)
    result_auc_base = pd.concat([result_auc_base, results_roc_auc_series_base], axis=1)

    openfe_model = OpenFEModel(data=data.data,
                                target=data.target,
                                numeric_features=data.numeric_features,
                                categorical_features=data.categorical_features)

    results_accuracy_series_openfe, results_roc_auc_series_openfe = openfe_model.fit_predict_evaluate()

    results_accuracy_series_openfe.name = dataset_name
    results_roc_auc_series_openfe.name = dataset_name

    result_ac_openfe = pd.concat([result_ac_openfe, results_accuracy_series_openfe], axis=1)
    result_auc_openfe = pd.concat([result_auc_openfe, results_roc_auc_series_openfe], axis=1)

    ifc_model = IFCModel(data=data.data,
                                target=data.target,
                                numeric_features=data.numeric_features,
                                categorical_features=data.categorical_features)

    results_accuracy_series_ifc, results_roc_auc_series_ifc = ifc_model.fit_predict_evaluate()

    results_accuracy_series_ifc.name = dataset_name
    results_roc_auc_series_ifc.name = dataset_name

    result_ac_ifc = pd.concat([result_ac_ifc, results_accuracy_series_ifc], axis=1)
    result_auc_ifc = pd.concat([result_auc_ifc, results_roc_auc_series_ifc], axis=1)

    al_model = ALModel(data=data.data,
                                target=data.target,
                                numeric_features=data.numeric_features,
                                categorical_features=data.categorical_features)

    results_accuracy_series_al, results_roc_auc_series_al = al_model.fit_predict_evaluate()

    results_accuracy_series_al.name = dataset_name
    results_roc_auc_series_al.name = dataset_name

    result_ac_al = pd.concat([result_ac_al, results_accuracy_series_al], axis=1)
    result_auc_al = pd.concat([result_auc_al, results_roc_auc_series_al], axis=1)

    al_mod_model = ALModModel(data=data.data,
                                target=data.target,
                                numeric_features=data.numeric_features,
                                categorical_features=data.categorical_features)

    results[dataset_name] = al_mod_model.fit_predict_evaluate()

    autofeat_model = AutoFeatModel(data=data.data,
                          target=data.target,
                          numeric_features=data.numeric_features,
                          categorical_features=data.categorical_features)
    accuracy_af, roc_auc_af = tpot_model.fit_predict_evaluate()
    
    results_accuracy_af[dataset_name] = accuracy_af
    results_roc_auc_af[dataset_name] = roc_auc_af
    
    tpot_model = TPOTModel(data=data.data,
                      target=data.target,
                      numeric_features=data.numeric_features,
                      categorical_features=data.categorical_features)
    accuracy_tpot, roc_auc_tpot = tpot_model.fit_predict_evaluate()
    
    results_accuracy_tpot[dataset_name] = accuracy_tpot
    results_roc_auc_tpot[dataset_name] = roc_auc_tpot

    safe_model = SAFEModel(data=data.data,
                      target=data.target,
                      numeric_features=data.numeric_features,
                      categorical_features=data.categorical_features)
    accuracy_safe, roc_auc_safe = safe_model.fit_predict_evaluate()
    
    results_accuracy_safe[dataset_name] = accuracy_safe
    results_roc_auc_safe[dataset_name] = roc_auc_safe


print('Base Result: Accurcay', result_ac_base, '\nRoc-Auc', result_auc_base)
print('\nOpenFE Result: Accurcay', result_ac_openfe, '\nRoc-Auc', result_auc_openfe)
print('\nIFC Result: Accurcay', result_ac_ifc, '\nRoc-Auc', result_auc_ifc)
print('\nAutoLeanrn Result: Accurcay', result_ac_al, '\nRoc-Auc', result_auc_al)
print('\nModified AuroLearn results: ', results)

df_accuracy = pd.DataFrame.from_dict(results_accuracy_safe, orient='index')
df_accuracy.index.name = 'Dataset'
df_accuracy.columns.name = 'Model'

df_roc_auc = pd.DataFrame.from_dict(results_roc_auc_safe, orient='index')
df_roc_auc.index.name = 'Dataset'
df_roc_auc.columns.name = 'Model'

print("\n SAFE Accuracy:")
print(df_accuracy)
print("\n SAFE ROC-AUC:")
print(df_roc_auc)

df_accuracy = pd.DataFrame.from_dict(results_accuracy_tpot, orient='index')
df_accuracy.index.name = 'Dataset'
df_accuracy.columns.name = 'Model'

df_roc_auc = pd.DataFrame.from_dict(results_roc_auc_tpot, orient='index')
df_roc_auc.index.name = 'Dataset'
df_roc_auc.columns.name = 'Model'

print("\n TPOT Accuracy:")
print(df_accuracy)
print("\n TPOT ROC-AUC:")
print(df_roc_auc)


df_accuracy = pd.DataFrame.from_dict(results_accuracy_af, orient='index')
df_accuracy.index.name = 'Dataset'
df_accuracy.columns.name = 'Model'

df_roc_auc = pd.DataFrame.from_dict(results_roc_auc_af, orient='index')
df_roc_auc.index.name = 'Dataset'
df_roc_auc.columns.name = 'Model'

print("\n AutoFeat Accuracy:")
print(df_accuracy)
print("\n AutoFeat ROC-AUC:")
print(df_roc_auc)