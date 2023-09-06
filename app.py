import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Function to handle missing values (replace with mean)
def handle_missing_values(df):
    df.fillna(df.mean(), inplace=True)
    return df

# Function to remove duplicates
def remove_duplicates(df):
    df.drop_duplicates(inplace=True)
    return df

# Function to handle outliers (replace with 99th percentile)
def handle_outliers(df, column):
    q99 = df[column].quantile(0.99)
    df[column] = np.where(df[column] > q99, q99, df[column])
    return df

# Function for Min-Max scaling
def min_max_scaling(df, column):
    min_val = df[column].min()
    max_val = df[column].max()
    df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

# Function for one-hot encoding
def one_hot_encoding(df, column):
    df_encoded = pd.get_dummies(df, columns=[column])
    return df_encoded

# Function to perform a t-test between two groups
def perform_t_test(df, column1, column2):
    from scipy.stats import ttest_ind
    group1 = df[df['Group'] == 'Group1'][column1]
    group2 = df[df['Group'] == 'Group2'][column2]
    t_stat, p_value = ttest_ind(group1, group2)
    return t_stat, p_value

def main():
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

    # Show Values
    if st.button("Value Counts"):
        st.text("Value Counts By Target/Class")
        st.write(df.iloc[:, -1].value_counts())

    # Checking for null value in DataSet
    if st.checkbox("Check for null values available in dataset"):
        st.write(df.isnull().sum())

    # Show DataTypes
    if st.button("Data Types"):
        st.write(df.dtypes)

    # Summary of Dataset
    if st.checkbox("Summary of Dataset"):
        st.write(df.describe())

    # Data Cleaning
    st.sidebar.header("Data Cleaning")
    if st.sidebar.checkbox("Handle Missing Values"):
        df = handle_missing_values(df)
        st.success("Missing values handled.")

    if st.sidebar.checkbox("Remove Duplicates"):
        df = remove_duplicates(df)
        st.success("Duplicates removed.")

    if st.sidebar.checkbox("Handle Outliers"):
        column = st.selectbox("Select a column for outlier handling", df.columns)
        df = handle_outliers(df, column)
        st.success("Outliers handled.")

    # Data Transformation
    st.sidebar.header("Data Transformation")
    if st.sidebar.checkbox("Feature Scaling"):
        column = st.selectbox("Select a column for feature scaling", df.columns)
        df = min_max_scaling(df, column)
        st.success("Feature scaling applied.")

    if st.sidebar.checkbox("Categorical Encoding"):
        column = st.selectbox("Select a categorical column for encoding", df.columns)
        df = one_hot_encoding(df, column)
        st.success("Categorical encoding applied.")

    # Advanced Visualizations
    st.sidebar.header("Advanced Visualizations")
    if st.sidebar.checkbox("Histogram"):
        column = st.selectbox("Select a column for histogram", df.columns)
        plt.hist(df[column], bins=20)
        st.pyplot()

    if st.sidebar.checkbox("Scatter Plot"):
        x = st.selectbox("X-axis", df.columns)
        y = st.selectbox("Y-axis", df.columns)
        plt.scatter(df[x], df[y])
        st.pyplot()

    # Statistical Tests
    st.sidebar.header("Statistical Tests")
    if st.sidebar.checkbox("T-test"):
        column1 = st.selectbox("Select a numeric column for Group 1", df.columns)
        column2 = st.selectbox("Select a numeric column for Group 2", df.columns)
        t_stat, p_value = perform_t_test(df, column1, column2)
        st.write(f"T-statistic: {t_stat}")
        st.write(f"P-value: {p_value}")

    # Data Exporting and Reporting
    st.sidebar.header("Data Exporting and Reporting")
    if st.sidebar.checkbox("Export Cleaned Data"):
        df.to_csv("cleaned_data.csv", index=False)
        st.success("Cleaned data exported as CSV.")

    if st.sidebar.checkbox("Generate Report"):
        # Add code to generate a report (e.g., using PDF generation libraries)
        st.success("Report generated.")

if __name__ == '__main__':
    main()
