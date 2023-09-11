import seaborn as sns
import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
import pandas_profiling
matplotlib.use('Agg')

def main():
    """ Machine Learning DataSet Explorer & Data Visualization """
    st.title("ML DataSet Explorer & Data Visualization")
    st.subheader("DataSet Explorer built with Streamlit")

    def file_selector(path='./datasets'):
        files = os.listdir(path)
        select_file = st.selectbox("Select a file(dataset)", files)
        return os.path.join(path, select_file)

    file = file_selector()
    st.info("You selected {}".format(file))

    # Read File .csv
    df = pd.read_csv(file)

    # Show Dataset
    if st.checkbox('Show Dataset'):
        st.write(df.head())

    # Show Shape of Dataset
    if st.checkbox("Shape of Dataset"):
        st.write(df.shape)

    # Perform EDA using pandas-profiling
    if st.checkbox("Perform EDA with pandas-profiling"):
        st.subheader("Exploratory Data Analysis with pandas-profiling")
        report = pandas_profiling.ProfileReport(df)
        st.write(report.to_html(), unsafe_allow_html=True)

    # ... (rest of your code)

if __name__ == '__main__':
    main()