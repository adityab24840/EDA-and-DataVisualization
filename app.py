import seaborn as sns
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
st.pyplot(plt.gcf())

# Set the page title and subheader
st.title("ML DataSet Explorer & Data Visualization")
st.subheader("DataSet Explorer built with Streamlit")

# Function to select a dataset file
def file_selector(path='./datasets'):
    files = os.listdir(path)
    select_file = st.selectbox("Select a file(dataset)", files)
    return os.path.join(path, select_file)

# Select a dataset file
file = file_selector()
st.info(f"You selected {file}")

# Read the selected dataset
@st.cache  # Cache the dataset to improve performance
def load_data(file):
    return pd.read_csv(file)

df = load_data(file)

# Sidebar - About and GitHub link
st.sidebar.header("About")
st.sidebar.info("Aditya N Bhatt")
st.sidebar.text("231057017")
st.sidebar.text("AI & ML")
st.sidebar.text("MSIS")

# Add your GitHub repository link here
github_repo = "https://github.com/adityab24840/EDA-and-DataVisualization/"
st.sidebar.markdown(f"[GitHub Repository]({github_repo})")

st.sidebar.text("Built with Streamlit")

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

# Show Summary
if st.checkbox("Summary of Dataset"):
    st.write(df.describe())

# Data Visualization
st.subheader("Data Visualization")

# Correlation Heatmap (Seaborn Plot)
if st.checkbox("Correlation Plot[Seaborn]"):
    st.write(sns.heatmap(df.corr(), annot=True))
    st.pyplot()

# Pie Chart
if st.checkbox("Pie Plot"):
    all_columns_names = df.columns.tolist()
    if st.button("Generate Pie Plot"):
        st.success("Generating A Pie Plot")
        st.write(df.iloc[:, -1].value_counts().plot.pie(autopct="%1.1f%%"))
        st.pyplot()

# Count Plot
if st.checkbox("Plot of Value Counts"):
    st.text("Value Counts By Target")
    all_columns_names = df.columns.tolist()
    primary_col = st.selectbox("Primary Columm to GroupBy", all_columns_names)
    selected_columns_names = st.multiselect("Select Columns", all_columns_names)
    if st.button("Plot"):
        st.text("Generate Plot")
        if selected_columns_names:
            vc_plot = df.groupby(primary_col)[selected_columns_names].count()
        else:
            vc_plot = df.iloc[:, -1].value_counts()
        st.write(vc_plot.plot(kind="bar"))
        st.pyplot()

# Customizable Plot
st.subheader("Customizable Plot")
all_columns_names = df.columns.tolist()
type_of_plot = st.selectbox("Select Type of Plot", ["area", "bar", "line", "hist", "box", "kde"])
selected_columns_names = st.multiselect("Select Columns To Plot", all_columns_names)

if st.button("Generate Plot"):
    st.success(f"Generating Customizable Plot of {type_of_plot} for {selected_columns_names}")

    # Plot By Streamlit
    if type_of_plot == 'area':
        cust_data = df[selected_columns_names]
        st.area_chart(cust_data)

    elif type_of_plot == 'bar':
        cust_data = df[selected_columns_names]
        st.bar_chart(cust_data)

    elif type_of_plot == 'line':
        cust_data = df[selected_columns_names]
        st.line_chart(cust_data)

    # Custom Plot
    elif type_of_plot:
        cust_plot = df[selected_columns_names].plot(kind=type_of_plot)
        st.write(cust_plot)
        st.pyplot()

# "Thanks" Button
if st.button("Thanks"):
    st.balloons()

