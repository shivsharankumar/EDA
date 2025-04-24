import streamlit as st
import pandas as pd
from utils import load_data, show_data_info

def data_collection():
    st.header("2. Data Collection")
    
    st.markdown("""
    ### Upload your dataset
    
    Upload your data in CSV or Excel format to begin the analysis process.
    """)
    
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        with st.spinner("Loading data..."):
            try:
                df = load_data(uploaded_file)
                if df is not None:
                    st.session_state.data = df
                    st.session_state.file_name = uploaded_file.name
                    st.success(f"Successfully loaded {uploaded_file.name} with {df.shape[0]} rows and {df.shape[1]} columns.")
                    
                    # Show data information
                    show_data_info(df)
                    
                    # Data source documentation
                    st.subheader("Document your data source")
                    data_source = st.text_area(
                        "Data Source Description", 
                        placeholder="Describe where this data comes from, how it was collected, and any relevant context...",
                        key="data_source_description"
                    )
                    
                    if st.button("Save Data Source Information"):
                        st.session_state.data_source_info = data_source
                        st.success("Data source information saved! You can now proceed to the Exploratory Data Analysis step.")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    else:
        if st.session_state.data is None:
            st.info("Please upload a CSV or Excel file to begin your analysis.")
            
            # Sample data option
            if st.button("Use Sample Dataset"):
                # Load a sample dataset
                df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
                st.session_state.data = df
                st.session_state.file_name = "iris.csv"
                st.success(f"Loaded sample Iris dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
                
                # Show data information
                show_data_info(df)
        else:
            st.success(f"Using previously loaded data: {st.session_state.file_name}")
            
            # Show data information for previously loaded data
            show_data_info(st.session_state.data)
            
            # Option to clear and upload new data
            if st.button("Clear data and upload new file"):
                st.session_state.data = None
                st.session_state.file_name = None
                st.session_state.cleaned_data = None
                st.session_state.transformed_data = None
                st.session_state.X_train = None
                st.session_state.X_test = None
                st.session_state.y_train = None
                st.session_state.y_test = None
                st.experimental_rerun()
    
    # Display tips
    with st.expander("Tips for Data Collection"):
        st.markdown("""
        - **Data quality** is crucial for successful analysis
        - **Document the data source** for future reference
        - **Understand the data context** before proceeding with analysis
        - **Consider data privacy** and compliance requirements
        - **Validate data integrity** by checking for inconsistencies
        """) 