import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import detect_outliers, get_download_link

def data_cleaning():
    st.header("4. Data Cleaning")
    
    if st.session_state.data is None:
        st.warning("No data available for cleaning. Please upload data in the Data Collection step.")
        return
    
    df = st.session_state.data.copy()
    
    st.markdown("""
    ### Clean Your Data
    
    Data cleaning is a crucial step to ensure your analysis is based on high-quality data.
    """)
    
    # Display tabs for different cleaning operations
    tabs = st.tabs(["Missing Values", "Duplicates", "Outliers", "Data Types", "Preview & Save"])
    
    # Initialize cleaned data if not already done
    if st.session_state.cleaned_data is None:
        st.session_state.cleaned_data = df.copy()
    
    cleaned_df = st.session_state.cleaned_data.copy()
    
    with tabs[0]:
        st.subheader("Handle Missing Values")
        
        # Show missing values info
        missing = pd.DataFrame(cleaned_df.isna().sum(), columns=['Missing Values'])
        missing['Percentage'] = (cleaned_df.isna().sum() / len(cleaned_df) * 100).round(2)
        missing = missing[missing['Missing Values'] > 0]
        
        if missing.empty:
            st.success("No missing values found in the dataset.")
        else:
            st.write("Columns with missing values:")
            st.dataframe(missing)
            
            st.write("Select columns to handle missing values:")
            
            for column in missing.index:
                st.write(f"### {column}: {missing.loc[column, 'Missing Values']} missing values ({missing.loc[column, 'Percentage']}%)")
                
                strategy = st.selectbox(f"Select strategy for {column}", 
                                      ["Keep missing", "Drop rows", "Replace with mean", 
                                       "Replace with median", "Replace with mode", 
                                       "Replace with custom value"],
                                      key=f"missing_{column}")
                
                if strategy == "Drop rows":
                    if st.button(f"Drop rows with missing values in {column}", key=f"drop_{column}"):
                        initial_shape = cleaned_df.shape
                        cleaned_df = cleaned_df.dropna(subset=[column])
                        st.success(f"Dropped {initial_shape[0] - cleaned_df.shape[0]} rows with missing values in {column}.")
                        st.session_state.cleaned_data = cleaned_df
                
                elif strategy == "Replace with mean":
                    if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                        if st.button(f"Replace with mean in {column}", key=f"mean_{column}"):
                            cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].mean())
                            st.success(f"Replaced missing values in {column} with mean.")
                            st.session_state.cleaned_data = cleaned_df
                    else:
                        st.warning(f"Cannot use mean for non-numeric column {column}.")
                
                elif strategy == "Replace with median":
                    if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                        if st.button(f"Replace with median in {column}", key=f"median_{column}"):
                            cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())
                            st.success(f"Replaced missing values in {column} with median.")
                            st.session_state.cleaned_data = cleaned_df
                    else:
                        st.warning(f"Cannot use median for non-numeric column {column}.")
                
                elif strategy == "Replace with mode":
                    if st.button(f"Replace with mode in {column}", key=f"mode_{column}"):
                        mode_value = cleaned_df[column].mode()[0]
                        cleaned_df[column] = cleaned_df[column].fillna(mode_value)
                        st.success(f"Replaced missing values in {column} with mode ({mode_value}).")
                        st.session_state.cleaned_data = cleaned_df
                
                elif strategy == "Replace with custom value":
                    custom_value = st.text_input(f"Enter custom value for {column}", key=f"custom_{column}")
                    
                    if st.button(f"Replace with custom value in {column}", key=f"custom_btn_{column}"):
                        # Convert custom value to the appropriate type
                        if pd.api.types.is_numeric_dtype(cleaned_df[column]):
                            try:
                                custom_value = float(custom_value)
                            except ValueError:
                                st.error(f"Invalid numeric value for {column}.")
                                continue
                        
                        cleaned_df[column] = cleaned_df[column].fillna(custom_value)
                        st.success(f"Replaced missing values in {column} with custom value ({custom_value}).")
                        st.session_state.cleaned_data = cleaned_df
    
    with tabs[1]:
        st.subheader("Handle Duplicates")
        
        # Check for duplicates
        duplicate_count = cleaned_df.duplicated().sum()
        
        if duplicate_count == 0:
            st.success("No duplicate rows found in the dataset.")
        else:
            st.warning(f"Found {duplicate_count} duplicate rows in the dataset.")
            
            if st.button("Show duplicate rows"):
                st.dataframe(cleaned_df[cleaned_df.duplicated(keep='first')])
            
            if st.button("Remove duplicate rows"):
                initial_shape = cleaned_df.shape
                cleaned_df = cleaned_df.drop_duplicates()
                st.success(f"Removed {initial_shape[0] - cleaned_df.shape[0]} duplicate rows.")
                st.session_state.cleaned_data = cleaned_df
    
    with tabs[2]:
        st.subheader("Handle Outliers")
        
        # Detect outliers
        outliers = detect_outliers(cleaned_df)
        
        if not outliers:
            st.success("No outliers detected using the IQR method.")
        else:
            st.write("Detected outliers in the following columns:")
            
            for column, info in outliers.items():
                st.write(f"### {column}: {info['count']} outliers ({info['percentage']:.2f}%)")
                
                # Plot the distribution with outlier bounds
                fig = px.histogram(cleaned_df, x=column)
                fig.add_vline(x=info['lower_bound'], line_dash="dash", line_color="red", annotation_text="Lower bound")
                fig.add_vline(x=info['upper_bound'], line_dash="dash", line_color="red", annotation_text="Upper bound")
                st.plotly_chart(fig, use_container_width=True)
                
                strategy = st.selectbox(f"Select strategy for outliers in {column}", 
                                      ["Keep outliers", "Remove outliers", "Cap outliers", "Replace with median"],
                                      key=f"outlier_{column}")
                
                if strategy == "Remove outliers":
                    if st.button(f"Remove outliers in {column}", key=f"remove_{column}"):
                        initial_shape = cleaned_df.shape
                        cleaned_df = cleaned_df[
                            (cleaned_df[column] >= info['lower_bound']) & 
                            (cleaned_df[column] <= info['upper_bound'])
                        ]
                        st.success(f"Removed {initial_shape[0] - cleaned_df.shape[0]} outliers in {column}.")
                        st.session_state.cleaned_data = cleaned_df
                
                elif strategy == "Cap outliers":
                    if st.button(f"Cap outliers in {column}", key=f"cap_{column}"):
                        cleaned_df[column] = cleaned_df[column].clip(info['lower_bound'], info['upper_bound'])
                        st.success(f"Capped outliers in {column}.")
                        st.session_state.cleaned_data = cleaned_df
                
                elif strategy == "Replace with median":
                    if st.button(f"Replace outliers with median in {column}", key=f"replace_{column}"):
                        median = cleaned_df[column].median()
                        mask = (cleaned_df[column] < info['lower_bound']) | (cleaned_df[column] > info['upper_bound'])
                        cleaned_df.loc[mask, column] = median
                        st.success(f"Replaced outliers in {column} with median.")
                        st.session_state.cleaned_data = cleaned_df
    
    with tabs[3]:
        st.subheader("Fix Data Types")
        
        # Show current data types
        dtypes = pd.DataFrame(cleaned_df.dtypes, columns=['Current Data Type'])
        dtypes.index.name = 'Column'
        st.dataframe(dtypes)
        
        # Allow changing data types
        col_to_change = st.selectbox("Select column to change data type", cleaned_df.columns)
        current_type = cleaned_df[col_to_change].dtype
        
        st.write(f"Current type of {col_to_change}: {current_type}")
        
        new_type = st.selectbox("Select new data type", 
                              ["int", "float", "str", "category", "datetime", "boolean"],
                              key="new_type")
        
        if st.button(f"Change data type of {col_to_change}"):
            try:
                if new_type == "int":
                    cleaned_df[col_to_change] = cleaned_df[col_to_change].astype(int)
                elif new_type == "float":
                    cleaned_df[col_to_change] = cleaned_df[col_to_change].astype(float)
                elif new_type == "str":
                    cleaned_df[col_to_change] = cleaned_df[col_to_change].astype(str)
                elif new_type == "category":
                    cleaned_df[col_to_change] = cleaned_df[col_to_change].astype('category')
                elif new_type == "datetime":
                    cleaned_df[col_to_change] = pd.to_datetime(cleaned_df[col_to_change])
                elif new_type == "boolean":
                    cleaned_df[col_to_change] = cleaned_df[col_to_change].astype(bool)
                
                st.success(f"Changed data type of {col_to_change} to {new_type}.")
                st.session_state.cleaned_data = cleaned_df
            except Exception as e:
                st.error(f"Error changing data type: {str(e)}")
    
    with tabs[4]:
        st.subheader("Preview Cleaned Data")
        
        # Show the cleaned data
        st.dataframe(cleaned_df.head(10))
        
        # Compare original and cleaned data
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("Original data shape:")
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        
        with col2:
            st.write("Cleaned data shape:")
            st.write(f"Rows: {cleaned_df.shape[0]}, Columns: {cleaned_df.shape[1]}")
        
        with col3:
            st.write("Difference:")
            st.write(f"Rows: {df.shape[0] - cleaned_df.shape[0]}, Columns: {df.shape[1] - cleaned_df.shape[1]}")
        
        # Save cleaned data
        if st.button("Finalize Cleaned Data"):
            st.session_state.cleaned_data = cleaned_df
            st.success("Cleaned data saved! You can now proceed to the Feature Engineering step.")
        
        # Option to download cleaned data
        st.markdown(get_download_link(cleaned_df, "cleaned_data.csv", "Download cleaned data as CSV"), unsafe_allow_html=True)
    
    # Display tips
    with st.expander("Tips for Data Cleaning"):
        st.markdown("""
        - **Handle missing values** appropriately based on their meaning in the context
        - **Remove duplicates** to avoid bias in your analysis
        - **Deal with outliers** only if they represent errors, not valid extreme values
        - **Correct data types** for better analysis and visualization
        - **Document your cleaning steps** for reproducibility
        """) 