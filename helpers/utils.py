import streamlit as st
import pandas as pd

def clicked(button):
    st.session_state.clicked[button] = True

def checkbox_clicked(button):
    st.session_state.checkbox_menu[button] = st.session_state.checkbox_menu[button] == False

def additional_clicked_fun(button):
    st.session_state.refreshed[button] += 1

@st.cache_data
def describe_dataframe(df):
    # Initialize a list to hold descriptions for each column
    column_descriptions = []
    
    for column in df.columns:
        # Basic column data
        col_type = df[column].dtype
        num_nulls = df[column].isnull().sum()
        null_info = "has some missing values" if num_nulls > 0 else "has no missing values"
        
        # Detailed stats for numeric columns
        if pd.api.types.is_numeric_dtype(df[column]):
            max_value = df[column].max()
            min_value = df[column].min()
            mean_value = df[column].mean()
            column_descriptions.append(f"{column} (numeric) - type: {col_type}, {null_info}, max: {max_value}, min: {min_value}, mean: {mean_value:.2f}")
        # Add more conditions for other data types (e.g., categorical, datetime) as needed
        else:
            column_descriptions.append(f"{column} - type: {col_type}, {null_info}")
    
    # Combine all column descriptions into a single string
    detailed_description = "; ".join(column_descriptions)
    
    overall_description = f"The dataset has {len(df)} rows and {len(df.columns)} columns. Column details: {detailed_description}."
    
    return overall_description

@st.cache_data
def to_show(df, show_selected, rows_to_show):
    switch_dic = {
        'First few rows': df.head(rows_to_show), 'Last few rows': df.tail(rows_to_show), 'Random':df.sample(rows_to_show)
    }
    st.write(f'There are {len(df)} rows and {len(df.columns)} columns.')
    # columns = [col for col in df.columns]
    # st.write('Column Names')
    # st.write(columns)

    return switch_dic[show_selected]
