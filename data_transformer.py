import pandas as pd

class DataTransformer:
    def __init__(self, file_path, target_column):
        self.data = pd.read_csv(file_path)
        self.target = target_column
        self.numeric_features = []
        self.categorical_features = []
        self.drop_na()
        self.identify_column_types()
        self.drop_target_column()

    def identify_column_types(self):
        for column in self.data.columns:
            if pd.api.types.is_numeric_dtype(self.data[column]) and self.data[column].dtype != 'object':
                self.numeric_features.append(column)
            elif pd.api.types.is_object_dtype(self.data[column]):
                self.categorical_features.append(column)

    def create_binary_class(self, column_name, positive_class):
        self.data[column_name] = self.data[column_name].apply(lambda x: 1 if x == positive_class else 0)

    def drop_columns(self, columns):
        self.data.drop(columns, axis=1, inplace=True)
        self.numeric_features = [col for col in self.numeric_features if col not in columns]
        self.categorical_features = [col for col in self.categorical_features if col not in columns]

    def drop_na(self):
        self.data.dropna(inplace=True)

    def drop_target_column(self):
        if self.target in self.numeric_features:
            self.numeric_features.remove(self.target)
        elif self.target in self.categorical_features:
            self.categorical_features.remove(self.target)

    def change_target(self, change):
        self.data[self.target] = self.data[self.target].apply(lambda x: 1 if x == change else 0)

    def get_target_distribution(self):
        distribution = self.data[self.target].value_counts(normalize=True) * 100
        percentafe_of_ones = distribution.get(1, 0)
        return percentafe_of_ones
