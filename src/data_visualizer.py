# data_visualizer.py
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

class DataVisualizer:
    def __init__(self, data):
        self.data = data

    def plot_correlation(self):
        st.write(sns.heatmap(self.data.corr(), annot=True))
        st.pyplot()

    def plot_pie_chart(self):
        all_columns_names = self.data.columns.tolist()
        if st.button("Generate Pie Plot"):
            st.success("Generating A Pie Plot")
            st.write(self.data.iloc[:, -1].value_counts().plot.pie(autopct="%1.1f%%"))
            st.pyplot()
