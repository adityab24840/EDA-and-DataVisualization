import seaborn as sns
import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pandas_profiling import ProfileReport

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

    # Show DataSet Column
    if st.checkbox("Select Columns to Show"):
        all_columns = df.columns.tolist()
        select_columns = st.multiselect("Select", all_columns)
        new_df = df[select_columns]
        st.dataframe(new_df)

    # Pandas Profiling Report
    if st.button("Generate Pandas Profiling Report"):
        st.text("Generating Pandas Profiling Report. This may take a while...")
        profile = ProfileReport(df, explorative=True)
        
        # Display the report using st.write()
        st.write(profile.to_widgets())

if __name__ == '__main__':
    main()
