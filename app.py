import os
import streamlit as st
from src.data_loader import DataLoader
from src.data_visualizer import DataVisualizer
from src.machine_learning_model import MachineLearningModel
from src.interactive_widgets import InteractiveWidgets
from src.about_Info import get_about_info

def main():
    st.title("ML DataSet Explorer & Data Visualization")
    st.set_option('deprecation.showPyplotGlobalUse', False)

    def file_selector(path='./datasets'):
        files = os.listdir(path)
        file_path = st.selectbox("Select a file(dataset)", files)
        return os.path.join(path, file_path)
    
    # Main content

   

    data_loader = DataLoader(file_path)
    data_loader.load_data()
    # Update the file path to include the 'datasets' folder
    file_path = os.path.join('./datasets', file_path)
    data_loader = DataLoader(file_path)
    data_loader.load_data()

    st.subheader("Data Preprocessing")
    preprocessing_method = st.selectbox("Select Data Preprocessing Method", ["None", "Handle Missing Values", "Encode Categorical Variables", "Scale/Normalize Features"])

    if preprocessing_method == "Handle Missing Values":
        missing_values_method = st.selectbox("Select Missing Value Handling Method", ["None", "Drop Rows", "Fill with Mean", "Fill with Median"])
        if missing_values_method != "None":
            data_loader.handle_missing_values(missing_values_method)

    if preprocessing_method == "Encode Categorical Variables":
        encoding_method = st.selectbox("Select Encoding Method", ["None", "Label Encoding", "One-Hot Encoding"])
        if encoding_method != "None":
            data_loader.encode_categorical_variables(encoding_method)

    if preprocessing_method == "Scale/Normalize Features":
        scaling_method = st.selectbox("Select Scaling Method", ["None", "Standardization (Z-score)", "Min-Max Scaling"])
        if scaling_method != "None":
            data_loader.scale_normalize_features(scaling_method)

    data_visualizer = DataVisualizer(data_loader.data)
    st.subheader("Data Visualization")
    visualization_method = st.selectbox("Select Visualization Method", ["None", "Correlation Plot", "Pie Chart"])

    if visualization_method == "Correlation Plot":
        data_visualizer.plot_correlation()

    if visualization_method == "Pie Chart":
        data_visualizer.plot_pie_chart()

    ml_model = MachineLearningModel(data_loader.data)
    st.subheader("Machine Learning Integration")
    if st.checkbox("Train Machine Learning Model"):
        target_column = st.selectbox("Select Target Column", data_loader.data.columns.tolist())
        ml_model.prepare_data(target_column)
        ml_model.train_model()
        ml_model.evaluate_model()

    interactive_widgets = InteractiveWidgets()
    st.subheader("Interactive Widgets")
    interactive_widgets.add_range_slider()

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
