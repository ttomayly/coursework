import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
from tpot import TPOTClassifier


class TPOTModel:
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

        model = TPOTClassifier(generations=2,
                               population_size=50,
                               verbosity=2,
                               random_state=42)


        pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model)])
      
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
       
        threshold = 0.5
        binary_predictions = [1 if pred >= threshold else 0 for pred in y_pred]

        accuracy = accuracy_score(y_test, binary_predictions)
        roc_auc = roc_auc_score(y_test, y_pred)

        return accuracy, roc_auc