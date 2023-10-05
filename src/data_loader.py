# data_loader.py
import pandas as pd
import streamlit as st
import os

class DataLoader:
    def __init__(self, file_path='./datasets'):
        self.file_path = file_path

    def load_data(self):
        st.info("You selected {}".format(self.file_path))
        self.df = pd.read_csv(self.file_path)
        self.numeric_columns = self.df.select_dtypes(include='number').columns.tolist()
        self.categorical_columns = self.df.select_dtypes(include='object').columns.tolist()

    def handle_missing_values(self, method):
        if method == "Drop Rows":
            self.df.dropna(inplace=True)
        elif method == "Fill with Mean":
            self.df.fillna(self.df.mean(), inplace=True)
        elif method == "Fill with Median":
            self.df.fillna(self.df.median(), inplace=True)
        # else:
            # Custom handling logic

    def encode_categorical_variables(self, encoding_method):
        if encoding_method == "Label Encoding":
            self.df[self.categorical_columns] = self.df[self.categorical_columns].apply(lambda col: pd.factorize(col)[0])
        else:
            self.df = pd.get_dummies(self.df, columns=self.categorical_columns)

    def scale_normalize_features(self, scaling_method):
        if scaling_method == "Standardization (Z-score)":
            self.df[self.numeric_columns] = (self.df[self.numeric_columns] - self.df[self.numeric_columns].mean()) / self.df[self.numeric_columns].std()
        else:
            self.df[self.numeric_columns] = (self.df[self.numeric_columns] - self.df[self.numeric_columns].min()) / (self.df[self.numeric_columns].max() - self.df[self.numeric_columns].min())

    def get_data(self):
        return self.df
