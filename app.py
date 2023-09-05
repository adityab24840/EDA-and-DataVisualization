import seaborn as sns
import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
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

    # html_temp = '''
    # <h1 style="text-align:center;">OR</h1>
    # '''
    # st.markdown(html_temp, unsafe_allow_html=True)
    # file = st.file_uploader("Upload a Dataset", type=['csv'])
    #
    # st.info("You selected {}".format(file))


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

    # Show Summary
    if st.checkbox("Summary of Dataset"):
        st.write(df.describe())

    # Plot and Visualization
    st.subheader("Data Visulaization")
    #Corelation
    #Seaborn Plot
    if st.checkbox("Correlation Plot[Seaborn]"):
        st.write(sns.heatmap(df.corr(),annot=True))
        st.pyplot()

    # Pie Chart
    if st.checkbox("Pie Plot"):
        all_columns_names = df.columns.tolist()
        if st.button("Generate Pie Plot"):
            st.success("Generating A Pie Plot")
            st.write(df.iloc[:,-1].value_counts().plot.pie(autopct="%1.1f%%"))
            st.pyplot()

    # Count Plot
    if st.checkbox("Plot of Value Counts"):
        st.text("Value Counts By Target")
        all_columns_names = df.columns.tolist()
        primary_col = st.selectbox("Primary Columm to GroupBy",all_columns_names)
        selected_columns_names = st.multiselect("Select Columns",all_columns_names)
        if st.button("Plot"):
            st.text("Generate Plot")
            if selected_columns_names:
                vc_plot = df.groupby(primary_col)[selected_columns_names].count()
            else:
                vc_plot = df.iloc[:,-1].value_counts()
            st.write(vc_plot.plot(kind="bar"))
            st.pyplot()

    # Customizable Plot
    st.subheader("Customizable Plot")
    all_columns_names = df.columns.tolist()
    type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
    selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

    if st.button("Generate Plot"):
        st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

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
            cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
            st.write(cust_plot)
            st.pyplot()

    if st.button("Thanks"):
        st.balloons()

    
    st.sidebar.header("About")
    st.sidebar.info("Mehul Anshumali")
    st.sidebar.text("Built with Streamlit")






if __name__ == '__main__':
    main()
