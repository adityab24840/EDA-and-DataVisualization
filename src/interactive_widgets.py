# interactive_widgets.py
import streamlit as st
max_value_of_data = 100  # Replace with the actual maximum value
min_value_of_data = 0  # Replace with the actual minimum value


class InteractiveWidgets:
    def add_range_slider(self):
        min_value = st.slider("Select Min Value", min_value_of_data, max_value_of_data)
        max_value = st.slider("Select Max Value", min_value_of_data, max_value_of_data)
        filtered_data = self.data[(self.data['numeric_column'] >= min_value) & (self.data['numeric_column'] <= max_value)]
        st.write(filtered_data)
