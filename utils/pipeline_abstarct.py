from abc import ABC, abstractmethod


class DataInput:
    @abstractmethod
    def get_input_dfs(self):
        pass

    @abstractmethod
    def compress_df(self):
        pass


class DataCleaner:
    def __init__(self, data_input):
        self.data_input = data_input

    @abstractmethod
    def correct_types(self):
        pass

    @abstractmethod
    def handle_na(self):
        pass

    def clean(self):
        self.clean_data = [x.copy() for x in self.data_input.get_dataframes()]
        self.correct_types()
        self.handle_na()

        return self.clean_data


class FeatureEngineering:
    def __init__(self, clean_data):
        self.clean_data = clean_data

    # This method should return a single dataframe containing every feature.
    @abstractmethod
    def transform(self, *args, **kwargs):
        pass


class PredictionModel:
    def __init__(self, data):
        self.data = data

    @abstractmethod
    def encode(self, data, preserve_order=True):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    def fit(self):
        # self.encoded_data = self.encode(self.data, preserve_order=False)
        pass

    @abstractmethod
    def is_score(self):
        pass

    @abstractmethod
    def produce_solution(self):
        pass

    def solve(self, test_data):
        self.encoded_test_data = self.encode(test_data)

    @abstractmethod
    def split_x_y(self, df):
        pass
