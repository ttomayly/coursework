from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from SafeTransformer import SafeTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


class SAFEModel:
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

        X_train_encoded = categorical_transformer.fit_transform(X_train[self.categorical_features])
        X_test_encoded = categorical_transformer.transform(X_test[self.categorical_features])

        X_train[categorical_transformer.get_feature_names_out()] = X_train_encoded
        X_test[categorical_transformer.get_feature_names_out()] = X_test_encoded

        X_train_scaled = numeric_transformer.fit_transform(X_train[self.numeric_features])
        X_test_scaled = numeric_transformer.transform(X_test[self.numeric_features])

        X_train[self.numeric_features] = X_train_scaled
        X_test[self.numeric_features] = X_test_scaled  

        return X_train, X_test, y_train, y_test

    
    def fit_predict_evaluate(self):
        X_train, X_test, y_train, y_test = self.preprocess_data()

        surrogate_model = GradientBoostingClassifier(n_estimators=100,
                            max_depth=4,
                            learning_rate=0.1,
                            loss='log_loss')
        surrogate_model = surrogate_model.fit(X_train, y_train)

        linear_model = LogisticRegression()

        safe_transformer = SafeTransformer(surrogate_model)
        pipeline = Pipeline(steps=[('safe', safe_transformer), ('linear', linear_model)])

        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        return accuracy, roc_auc