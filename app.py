import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport

# Additional libraries for data preprocessing and machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    """ Machine Learning DataSet Explorer & Data Visualization """
    st.title("ML DataSet Explorer & Data Visualization")
    st.subheader("DataSet Explorer and Model Builder with Streamlit")

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

    # Data Preprocessing Section
    st.subheader("Data Preprocessing")
    target_col = st.selectbox("Select Target Column", all_columns)

    # Handle missing values
    if st.checkbox("Handle Missing Values"):
        df.fillna(df.mean(), inplace=True)  # Example: Fill missing values with mean

    # Encode categorical variables
    if st.checkbox("Encode Categorical Variables"):
        cat_cols = [col for col in df.columns if df[col].dtype == 'object']
        if cat_cols:
            le = LabelEncoder()
            for col in cat_cols:
                df[col] = le.fit_transform(df[col])

    # Model Building Section
    st.subheader("Model Building")

    # Select features
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols]
    y = df[target_col]

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a random forest classifier
    if st.checkbox("Train Model"):
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Model Accuracy: {accuracy:.2f}")

        # Display classification report
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

    # Data Visualization Section
    st.subheader("Data Visualization")

    # Correlation Plot
    if st.checkbox("Correlation Plot[Seaborn]"):
        st.write(sns.heatmap(df.corr(), annot=True))
        st.pyplot()

    # Histograms for numerical columns
    if st.checkbox("Histograms"):
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numerical_cols:
            st.subheader(f"Histogram for {col}")
            plt.hist(df[col], bins=20)
            st.pyplot()

    # Pie Chart for categorical target
    if df[target_col].dtype == 'object':
        if st.checkbox("Pie Plot"):
            st.write(df[target_col].value_counts().plot.pie(autopct="%1.1f%%"))
            st.pyplot()

    # Customizable Plot
    st.subheader("Customizable Plot")
    type_of_plot = st.selectbox("Select Type of Plot", ["area", "bar", "line", "hist", "box", "kde"])
    selected_columns_names = st.multiselect("Select Columns To Plot", all_columns)

    if st.button("Generate Plot"):
        st.success(f"Generating Customizable Plot of {type_of_plot} for {selected_columns_names}")
        if selected_columns_names:
            if type_of_plot == 'area':
                st.area_chart(df[selected_columns_names])
            elif type_of_plot == 'bar':
                st.bar_chart(df[selected_columns_names])
            elif type_of_plot == 'line':
                st.line_chart(df[selected_columns_names])
            elif type_of_plot == 'hist':
                st.write(df[selected_columns_names].hist(bins=20))
                st.pyplot()
            elif type_of_plot == 'box':
                st.write(df[selected_columns_names].plot(kind='box'))
                st.pyplot()
            elif type_of_plot == 'kde':
                st.write(df[selected_columns_names].plot(kind='kde'))
                st.pyplot()

    if st.button("Thanks"):
        st.balloons()

    st.sidebar.header("About")
    st.sidebar.info("Aditya N Bhatt")
    st.sidebar.text("231057017")
    st.sidebar.text("AI & ML")
    st.sidebar.text("MSIS")
    github_repo = "https://github.com/adityab24840/EDA-and-DataVisualization/"
    st.sidebar.markdown(f"[GitHub Repository]({github_repo})")
    st.sidebar.text("Built with Streamlit")

if __name__ == '__main__':
    main()
