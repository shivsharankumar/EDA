import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def data_splitting():
    st.header("6. Data Splitting")
    
    if st.session_state.transformed_data is None:
        if st.session_state.cleaned_data is None:
            if st.session_state.data is None:
                st.warning("No data available for splitting. Please upload data in the Data Collection step.")
                return
            else:
                st.warning("Please clean and transform your data before proceeding to data splitting.")
                df = st.session_state.data.copy()
        else:
            st.warning("Please transform your data before proceeding to data splitting.")
            df = st.session_state.cleaned_data.copy()
    else:
        df = st.session_state.transformed_data.copy()
    
    st.markdown("""
    ### Split Your Data
    
    Splitting your data into training, validation, and test sets is crucial for model development and evaluation.
    """)
    
    st.subheader("Select Target Variable")
    target_variable = st.selectbox("Select your target variable (what you want to predict)", df.columns)
    
    st.subheader("Select Features")
    feature_columns = st.multiselect(
        "Select features to include in the model", 
        [col for col in df.columns if col != target_variable],
        default=[col for col in df.columns if col != target_variable]
    )
    
    if not feature_columns:
        st.warning("Please select at least one feature.")
        return
    
    # Preview the data
    st.subheader("Preview Features and Target")
    preview_df = df[[target_variable] + feature_columns].head(5)
    st.dataframe(preview_df)
    
    # Splitting options
    st.subheader("Splitting Options")
    
    split_type = st.radio(
        "Select split type",
        ["Train-Test Split", "Train-Validation-Test Split"]
    )
    
    if split_type == "Train-Test Split":
        test_size = st.slider("Test size (%)", 10, 40, 20)
        test_size = test_size / 100  # Convert to fraction
        
        random_state = st.number_input("Random state (for reproducibility)", value=42)
        
        # Check if target is categorical for stratification
        is_categorical = False
        if df[target_variable].dtype == 'object' or df[target_variable].dtype.name == 'category' or df[target_variable].nunique() < 10:
            is_categorical = True
            stratify_option = st.checkbox("Use stratified split (recommended for classification tasks)", value=True)
        else:
            stratify_option = False
        
        if st.button("Split Data"):
            try:
                X = df[feature_columns]
                y = df[target_variable]
                
                if stratify_option:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state, stratify=y
                    )
                    st.success("Data split with stratification.")
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )
                    st.success("Data split successfully.")
                
                # Store in session state
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.feature_columns = feature_columns
                st.session_state.target_variable = target_variable
                
                # Display split information
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Training set:")
                    st.write(f"X_train shape: {X_train.shape}")
                    st.write(f"y_train shape: {y_train.shape}")
                
                with col2:
                    st.write("Test set:")
                    st.write(f"X_test shape: {X_test.shape}")
                    st.write(f"y_test shape: {y_test.shape}")
                
                # Display target distribution
                if is_categorical:
                    st.subheader("Target Distribution in Splits")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Training set target distribution:")
                        train_dist = y_train.value_counts(normalize=True) * 100
                        train_dist = train_dist.rename("Percentage (%)").reset_index()
                        st.dataframe(train_dist)
                    
                    with col2:
                        st.write("Test set target distribution:")
                        test_dist = y_test.value_counts(normalize=True) * 100
                        test_dist = test_dist.rename("Percentage (%)").reset_index()
                        st.dataframe(test_dist)
                
                st.success("Data split successfully saved. You can now proceed to checking for imbalanced data.")
                
            except Exception as e:
                st.error(f"Error splitting data: {str(e)}")
    
    else:  # Train-Validation-Test Split
        test_size = st.slider("Test size (%)", 10, 30, 20)
        val_size = st.slider("Validation size (%)", 10, 30, 20)
        
        total_size = test_size + val_size
        if total_size >= 60:
            st.warning("Warning: Train set will be less than 40% of the data.")
        
        test_size = test_size / 100  # Convert to fraction
        val_size = val_size / 100  # Convert to fraction
        
        random_state = st.number_input("Random state (for reproducibility)", value=42)
        
        # Check if target is categorical for stratification
        is_categorical = False
        if df[target_variable].dtype == 'object' or df[target_variable].dtype.name == 'category' or df[target_variable].nunique() < 10:
            is_categorical = True
            stratify_option = st.checkbox("Use stratified split (recommended for classification tasks)", value=True)
        else:
            stratify_option = False
        
        if st.button("Split Data"):
            try:
                X = df[feature_columns]
                y = df[target_variable]
                
                # First split: train + validation vs test
                if stratify_option:
                    X_temp, X_test, y_temp, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state, stratify=y
                    )
                    
                    # Second split: train vs validation
                    # Adjust validation size based on the size of X_temp
                    val_size_adjusted = val_size / (1 - test_size)
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_temp, y_temp, test_size=val_size_adjusted, 
                        random_state=random_state, stratify=y_temp
                    )
                    
                    st.success("Data split with stratification.")
                else:
                    X_temp, X_test, y_temp, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )
                    
                    # Second split: train vs validation
                    val_size_adjusted = val_size / (1 - test_size)
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_temp, y_temp, test_size=val_size_adjusted, 
                        random_state=random_state
                    )
                    
                    st.success("Data split successfully.")
                
                # Store in session state
                st.session_state.X_train = X_train
                st.session_state.X_val = X_val
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_val = y_val
                st.session_state.y_test = y_test
                st.session_state.feature_columns = feature_columns
                st.session_state.target_variable = target_variable
                
                # Display split information
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("Training set:")
                    st.write(f"X_train shape: {X_train.shape}")
                    st.write(f"y_train shape: {y_train.shape}")
                
                with col2:
                    st.write("Validation set:")
                    st.write(f"X_val shape: {X_val.shape}")
                    st.write(f"y_val shape: {y_val.shape}")
                
                with col3:
                    st.write("Test set:")
                    st.write(f"X_test shape: {X_test.shape}")
                    st.write(f"y_test shape: {y_test.shape}")
                
                # Display target distribution
                if is_categorical:
                    st.subheader("Target Distribution in Splits")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("Training set target distribution:")
                        train_dist = y_train.value_counts(normalize=True) * 100
                        train_dist = train_dist.rename("Percentage (%)").reset_index()
                        st.dataframe(train_dist)
                    
                    with col2:
                        st.write("Validation set target distribution:")
                        val_dist = y_val.value_counts(normalize=True) * 100
                        val_dist = val_dist.rename("Percentage (%)").reset_index()
                        st.dataframe(val_dist)
                    
                    with col3:
                        st.write("Test set target distribution:")
                        test_dist = y_test.value_counts(normalize=True) * 100
                        test_dist = test_dist.rename("Percentage (%)").reset_index()
                        st.dataframe(test_dist)
                
                st.success("Data split successfully saved. You can now proceed to checking for imbalanced data.")
                
            except Exception as e:
                st.error(f"Error splitting data: {str(e)}")
    
    # Display tips
    with st.expander("Tips for Data Splitting"):
        st.markdown("""
        - **Train-test split** is essential to evaluate model performance on unseen data
        - **Validation set** helps tune hyperparameters without leaking information from the test set
        - **Stratified sampling** preserves the class distribution in classification tasks
        - **Use a fixed random state** to ensure reproducibility
        - **K-fold cross-validation** can be an alternative to a single validation split
        """) 