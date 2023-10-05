import seaborn as sns
import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from about_Info import get_about_info
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from scipy.stats import ttest_1samp

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
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(include='object').columns.tolist()

    # Show Dataset
    if st.checkbox('Show Dataset'):
        st.write(df.head())

    # Show Shape of Dataset
    if st.checkbox("Shape of Dataset"):
        st.write(df.shape)

    # Data Preprocessing
    st.subheader("Data Preprocessing")
    if st.checkbox("Handle Missing Values"):
        selected_method = st.selectbox("Select Method", ["Drop Rows", "Fill with Mean", "Fill with Median", "Custom"])
        if selected_method == "Drop Rows":
            df.dropna(inplace=True)
        elif selected_method == "Fill with Mean":
            df.fillna(df.mean(), inplace=True)
        elif selected_method == "Fill with Median":
            df.fillna(df.median(), inplace=True)
       # else:
            # Custom handling logic

    if st.checkbox("Encode Categorical Variables"):
        encoding_method = st.selectbox("Select Encoding Method", ["Label Encoding", "One-Hot Encoding"])
        if encoding_method == "Label Encoding":
            df[categorical_columns] = df[categorical_columns].apply(LabelEncoder().fit_transform)
        else:
            df = pd.get_dummies(df, columns=categorical_columns)

    if st.checkbox("Scale/Normalize Features"):
        scaling_method = st.selectbox("Select Scaling Method", ["Standardization (Z-score)", "Min-Max Scaling"])
        if scaling_method == "Standardization (Z-score)":
            df[numeric_columns] = StandardScaler().fit_transform(df[numeric_columns])
        else:
            df[numeric_columns] = MinMaxScaler().fit_transform(df[numeric_columns])

    # Feature Selection and Engineering
    st.subheader("Feature Selection and Engineering")
    if st.checkbox("Feature Selection"):
        selected_features = st.multiselect("Select Features", numeric_columns)
        df = df[selected_features]

    if st.checkbox("Feature Engineering"):
        # Example: Create a new feature 'total' by summing two existing features
        df['total'] = df['feature1'] + df['feature2']

    # Time Series Analysis
    st.subheader("Time Series Analysis")
    if st.checkbox("Time Series Analysis"):
        if 'datetime_column' not in df.columns:
            st.warning("No datetime column found in the dataset.")
        else:
            df['datetime_column'] = pd.to_datetime(df['datetime_column'])
            df.set_index('datetime_column', inplace=True)

            resampling_method = st.selectbox("Select Resampling Method", ["D", "W", "M", "Q", "Y"])
            resampled_df = df.resample(resampling_method).mean()

            st.line_chart(resampled_df)

    # Machine Learning Integration
    st.subheader("Machine Learning Integration")
    if st.checkbox("Machine Learning Integration"):
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        X = df.drop("target", axis=1)
        y = df["target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Model Accuracy: {accuracy}")

    # Interactive Widgets
    st.subheader("Interactive Widgets")
    if st.checkbox("Interactive Widgets"):
        min_value = st.slider("Select Min Value", min_value_of_data, max_value_of_data)
        max_value = st.slider("Select Max Value", min_value_of_data, max_value_of_data)
        filtered_data = df[(df['numeric_column'] >= min_value) & (df['numeric_column'] <= max_value)]
        st.write(filtered_data)

    # Advanced Visualization
    st.subheader("Advanced Visualization")
    if st.checkbox("Advanced Visualization"):
        # Example: Create an interactive 3D scatter plot with Plotly
        import plotly.express as px
        fig = px.scatter_3d(df, x='feature1', y='feature2', z='feature3', color='target')
        st.plotly_chart(fig)

    # Data Export
    st.subheader("Data Export")
    if st.checkbox("Export Data"):
        export_format = st.selectbox("Select Export Format", ["CSV", "Excel"])
        if export_format == "CSV":
            st.write("Exporting data to CSV format...")
            df.to_csv("exported_data.csv", index=False)
            st.success("Data exported successfully as CSV.")
        elif export_format == "Excel":
            st.write("Exporting data to Excel format...")
            df.to_excel("exported_data.xlsx", index=False)
            st.success("Data exported successfully as Excel.")

    # Model Evaluation Metrics
    st.subheader("Model Evaluation Metrics")
   # if st.checkbox("Model Evaluation"):
        # Calculate and display model evaluation metrics
        # Example: accuracy, precision, recall, F1-score
        # Insert your code here for model evaluation

    # Hyperparameter Tuning
    st.subheader("Hyperparameter Tuning")
    #if st.checkbox("Hyperparameter Tuning"):
        # Implement hyperparameter tuning for machine learning models
        # Insert your code here for hyperparameter tuning

    # Exploratory Data Analysis (EDA)
    st.subheader("Exploratory Data Analysis (EDA)")
    #if st.checkbox("Exploratory Data Analysis"):
        # Implement additional EDA functionalities
        # Insert your code here for EDA

    # Customized Plots
    st.subheader("Customized Plots")
    #if st.checkbox("Customized Plots"):
        # Allow users to customize plot styles, labels, and colors
        # Insert your code here for customized plots

    # Dashboard Layout
    st.subheader("Dashboard Layout")
    #if st.checkbox("Dashboard Layout"):
        # Create an organized dashboard layout with tabs or sections
        # Insert your code here for dashboard layout

    # User Authentication
    st.subheader("User Authentication")
    #if st.checkbox("User Authentication"):
        # Implement user authentication for personalization
        # Insert your code here for user authentication

    # Documentation and Help
    st.subheader("Documentation and Help")
    #if st.checkbox("Documentation"):
        # Provide documentation or tooltips for user guidance

    # Display About Information in Sidebar Area
    st.sidebar.header("About")
    about_info = get_about_info()
    st.sidebar.write(f"Name: {about_info['Name']}")
    st.sidebar.write(f"Student ID: {about_info['Student ID']}")
    st.sidebar.write(f"Major: {about_info['Major']}")
    st.sidebar.write(f"College: {about_info['College']}")
    st.sidebar.text("2023-25")
    st.sidebar.markdown(f"GitHub Repository: [GitHub Repo]({about_info['GitHub Repo']})")
    st.sidebar.text("Built with Streamlit")

if __name__ == '__main__':
    main()
