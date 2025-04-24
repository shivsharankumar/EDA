import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from utils import get_download_link

def feature_engineering():
    st.header("5. Feature Engineering")
    
    if st.session_state.cleaned_data is None:
        if st.session_state.data is None:
            st.warning("No data available for feature engineering. Please upload data in the Data Collection step.")
            return
        else:
            st.warning("Please clean your data before proceeding to feature engineering.")
            st.session_state.cleaned_data = st.session_state.data.copy()
    
    df = st.session_state.cleaned_data.copy()
    
    # Initialize transformed data if not already done
    if st.session_state.transformed_data is None:
        st.session_state.transformed_data = df.copy()
    
    transformed_df = st.session_state.transformed_data.copy()
    
    st.markdown("""
    ### Transform Your Features
    
    Feature engineering is the process of creating new features or transforming existing ones to improve model performance.
    """)
    
    # Display tabs for different feature engineering operations
    tabs = st.tabs([
        "Feature Creation", 
        "Encoding Categorical", 
        "Scaling/Normalization", 
        "Dimensionality Reduction",
        "Preview & Save"
    ])
    
    with tabs[0]:
        st.subheader("Create New Features")
        
        st.markdown("""
        Create new features from existing ones. This can include:
        - Mathematical transformations (log, square root, etc.)
        - Combining features
        - Date/time extraction (month, day of week, etc.)
        - Text feature extraction
        """)
        
        # Feature transformations
        st.write("### Mathematical Transformations")
        
        numerical_cols = transformed_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if numerical_cols:
            col_to_transform = st.selectbox("Select column to transform", numerical_cols, key="transform_col")
            
            transform_type = st.selectbox("Select transformation", [
                "None", "Log", "Square Root", "Square", "Cube", "Z-score", 
                "Min-Max", "Binning"
            ], key="transform_type")
            
            if transform_type != "None":
                new_col_name = st.text_input("New column name", f"{col_to_transform}_{transform_type.lower().replace(' ', '_')}")
                
                if st.button("Apply Transformation"):
                    try:
                        if transform_type == "Log":
                            # Handle negative or zero values
                            min_val = transformed_df[col_to_transform].min()
                            if min_val <= 0:
                                offset = abs(min_val) + 1
                                transformed_df[new_col_name] = np.log(transformed_df[col_to_transform] + offset)
                                st.info(f"Added {offset} to all values before log transformation to handle non-positive values.")
                            else:
                                transformed_df[new_col_name] = np.log(transformed_df[col_to_transform])
                            
                        elif transform_type == "Square Root":
                            # Handle negative values
                            min_val = transformed_df[col_to_transform].min()
                            if min_val < 0:
                                offset = abs(min_val)
                                transformed_df[new_col_name] = np.sqrt(transformed_df[col_to_transform] + offset)
                                st.info(f"Added {offset} to all values before square root to handle negative values.")
                            else:
                                transformed_df[new_col_name] = np.sqrt(transformed_df[col_to_transform])
                            
                        elif transform_type == "Square":
                            transformed_df[new_col_name] = transformed_df[col_to_transform] ** 2
                            
                        elif transform_type == "Cube":
                            transformed_df[new_col_name] = transformed_df[col_to_transform] ** 3
                            
                        elif transform_type == "Z-score":
                            mean = transformed_df[col_to_transform].mean()
                            std = transformed_df[col_to_transform].std()
                            transformed_df[new_col_name] = (transformed_df[col_to_transform] - mean) / std
                            
                        elif transform_type == "Min-Max":
                            min_val = transformed_df[col_to_transform].min()
                            max_val = transformed_df[col_to_transform].max()
                            transformed_df[new_col_name] = (transformed_df[col_to_transform] - min_val) / (max_val - min_val)
                            
                        elif transform_type == "Binning":
                            num_bins = st.slider("Number of bins", 2, 20, 5)
                            transformed_df[new_col_name] = pd.qcut(transformed_df[col_to_transform], q=num_bins, labels=False)
                        
                        st.success(f"Created new feature: {new_col_name}")
                        
                        # Show a preview of the transformed feature
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("Original Feature Distribution")
                            fig = px.histogram(transformed_df, x=col_to_transform)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.write("Transformed Feature Distribution")
                            fig = px.histogram(transformed_df, x=new_col_name)
                            st.plotly_chart(fig, use_container_width=True)
                            
                        st.session_state.transformed_data = transformed_df
                    except Exception as e:
                        st.error(f"Error applying transformation: {str(e)}")
        else:
            st.info("No numerical columns found for transformation.")
        
        # Date feature extraction
        st.write("### Date Feature Extraction")
        
        datetime_cols = []
        for col in transformed_df.columns:
            if pd.api.types.is_datetime64_any_dtype(transformed_df[col]):
                datetime_cols.append(col)
            elif transformed_df[col].dtype == 'object':
                # Try to convert to datetime
                try:
                    pd.to_datetime(transformed_df[col])
                    datetime_cols.append(col)
                except:
                    pass
        
        if datetime_cols:
            date_col = st.selectbox("Select date column", datetime_cols, key="date_col")
            
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(transformed_df[date_col]):
                if st.button("Convert to datetime"):
                    try:
                        transformed_df[date_col] = pd.to_datetime(transformed_df[date_col])
                        st.success(f"Converted {date_col} to datetime.")
                        st.session_state.transformed_data = transformed_df
                        datetime_cols = [date_col]  # Reset to only use the converted column
                    except Exception as e:
                        st.error(f"Error converting to datetime: {str(e)}")
                        datetime_cols = []
            
            if datetime_cols and pd.api.types.is_datetime64_any_dtype(transformed_df[date_col]):
                date_features = st.multiselect("Select date features to extract", [
                    "Year", "Month", "Day", "Day of Week", "Hour", "Minute", "Second",
                    "Quarter", "Is Weekend", "Week of Year"
                ], key="date_features")
                
                if date_features and st.button("Extract Date Features"):
                    for feature in date_features:
                        try:
                            if feature == "Year":
                                transformed_df[f"{date_col}_year"] = transformed_df[date_col].dt.year
                            elif feature == "Month":
                                transformed_df[f"{date_col}_month"] = transformed_df[date_col].dt.month
                            elif feature == "Day":
                                transformed_df[f"{date_col}_day"] = transformed_df[date_col].dt.day
                            elif feature == "Day of Week":
                                transformed_df[f"{date_col}_dayofweek"] = transformed_df[date_col].dt.dayofweek
                            elif feature == "Hour":
                                transformed_df[f"{date_col}_hour"] = transformed_df[date_col].dt.hour
                            elif feature == "Minute":
                                transformed_df[f"{date_col}_minute"] = transformed_df[date_col].dt.minute
                            elif feature == "Second":
                                transformed_df[f"{date_col}_second"] = transformed_df[date_col].dt.second
                            elif feature == "Quarter":
                                transformed_df[f"{date_col}_quarter"] = transformed_df[date_col].dt.quarter
                            elif feature == "Is Weekend":
                                transformed_df[f"{date_col}_is_weekend"] = transformed_df[date_col].dt.dayofweek >= 5
                            elif feature == "Week of Year":
                                transformed_df[f"{date_col}_weekofyear"] = transformed_df[date_col].dt.isocalendar().week
                        except Exception as e:
                            st.error(f"Error extracting {feature}: {str(e)}")
                    
                    st.success(f"Extracted date features from {date_col}.")
                    st.session_state.transformed_data = transformed_df
        else:
            st.info("No datetime columns found for feature extraction.")
        
        # Interaction features
        st.write("### Feature Interactions")
        
        numerical_cols = transformed_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if len(numerical_cols) >= 2:
            col1 = st.selectbox("Select first feature", numerical_cols, key="interaction_col1")
            col2 = st.selectbox("Select second feature", numerical_cols, key="interaction_col2", index=1 if len(numerical_cols) > 1 else 0)
            
            interaction_type = st.selectbox("Select interaction type", [
                "Multiplication", "Addition", "Subtraction", "Division", "Ratio"
            ], key="interaction_type")
            
            new_col_name = st.text_input("New column name", f"{col1}_{interaction_type.lower()}_{col2}")
            
            if st.button("Create Interaction Feature"):
                try:
                    if interaction_type == "Multiplication":
                        transformed_df[new_col_name] = transformed_df[col1] * transformed_df[col2]
                    elif interaction_type == "Addition":
                        transformed_df[new_col_name] = transformed_df[col1] + transformed_df[col2]
                    elif interaction_type == "Subtraction":
                        transformed_df[new_col_name] = transformed_df[col1] - transformed_df[col2]
                    elif interaction_type == "Division":
                        # Handle division by zero
                        if (transformed_df[col2] == 0).any():
                            transformed_df[new_col_name] = transformed_df[col1] / (transformed_df[col2] + 1e-10)
                            st.info("Added small constant to divisor to avoid division by zero.")
                        else:
                            transformed_df[new_col_name] = transformed_df[col1] / transformed_df[col2]
                    elif interaction_type == "Ratio":
                        # Calculate the ratio relative to sum
                        sum_cols = transformed_df[col1] + transformed_df[col2]
                        transformed_df[new_col_name] = transformed_df[col1] / (sum_cols + 1e-10)
                    
                    st.success(f"Created interaction feature: {new_col_name}")
                    st.session_state.transformed_data = transformed_df
                except Exception as e:
                    st.error(f"Error creating interaction feature: {str(e)}")
        else:
            st.info("Need at least two numerical columns to create interaction features.")
    
    with tabs[1]:
        st.subheader("Encode Categorical Features")
        
        categorical_cols = transformed_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if categorical_cols:
            st.write("### Select Categorical Features to Encode")
            
            col_to_encode = st.selectbox("Select column to encode", categorical_cols, key="encode_col")
            
            # Show value counts for the selected column
            value_counts = transformed_df[col_to_encode].value_counts()
            st.write(f"Unique values in {col_to_encode}:")
            st.dataframe(value_counts)
            
            encoding_type = st.selectbox("Select encoding method", [
                "One-Hot Encoding", "Label Encoding", "Ordinal Encoding"
            ], key="encoding_type")
            
            if encoding_type == "Ordinal Encoding":
                # For ordinal encoding, need to specify the order
                unique_values = transformed_df[col_to_encode].unique().tolist()
                st.write("Drag to reorder values (from lowest to highest):")
                ordered_values = st.multiselect("Order of values", unique_values, default=unique_values)
                
                if len(ordered_values) != len(unique_values):
                    st.warning("Please include all unique values in the ordering.")
                
                if st.button("Apply Ordinal Encoding") and len(ordered_values) == len(unique_values):
                    try:
                        # Create a mapping dictionary
                        mapping = {val: i for i, val in enumerate(ordered_values)}
                        transformed_df[f"{col_to_encode}_ordinal"] = transformed_df[col_to_encode].map(mapping)
                        
                        st.success(f"Applied ordinal encoding to {col_to_encode}.")
                        st.session_state.transformed_data = transformed_df
                    except Exception as e:
                        st.error(f"Error applying ordinal encoding: {str(e)}")
            
            elif encoding_type == "One-Hot Encoding":
                drop_first = st.checkbox("Drop first category (to avoid multicollinearity)", value=True)
                
                if st.button("Apply One-Hot Encoding"):
                    try:
                        # Apply one-hot encoding
                        dummies = pd.get_dummies(transformed_df[col_to_encode], prefix=col_to_encode, drop_first=drop_first)
                        
                        # Add the new columns to the dataframe
                        transformed_df = pd.concat([transformed_df, dummies], axis=1)
                        
                        st.success(f"Applied one-hot encoding to {col_to_encode}. Added {dummies.shape[1]} new columns.")
                        st.session_state.transformed_data = transformed_df
                    except Exception as e:
                        st.error(f"Error applying one-hot encoding: {str(e)}")
            
            elif encoding_type == "Label Encoding":
                if st.button("Apply Label Encoding"):
                    try:
                        # Apply label encoding
                        le = LabelEncoder()
                        transformed_df[f"{col_to_encode}_label"] = le.fit_transform(transformed_df[col_to_encode])
                        
                        # Show the mapping
                        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                        st.write("Label Encoding Mapping:")
                        st.json(mapping)
                        
                        st.success(f"Applied label encoding to {col_to_encode}.")
                        st.session_state.transformed_data = transformed_df
                    except Exception as e:
                        st.error(f"Error applying label encoding: {str(e)}")
        else:
            st.info("No categorical columns found for encoding.")
    
    with tabs[2]:
        st.subheader("Scale/Normalize Features")
        
        numerical_cols = transformed_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if numerical_cols:
            st.write("### Select Features to Scale")
            
            features_to_scale = st.multiselect("Select columns to scale", numerical_cols, key="scale_cols")
            
            scaling_method = st.selectbox("Select scaling method", [
                "Standard Scaling (Z-score)", 
                "Min-Max Scaling", 
                "Robust Scaling", 
                "Log Transformation"
            ], key="scaling_method")
            
            if features_to_scale and st.button("Apply Scaling"):
                try:
                    if scaling_method == "Standard Scaling (Z-score)":
                        scaler = StandardScaler()
                        transformed_df[features_to_scale] = scaler.fit_transform(transformed_df[features_to_scale])
                        st.success("Applied standard scaling.")
                        
                    elif scaling_method == "Min-Max Scaling":
                        scaler = MinMaxScaler()
                        transformed_df[features_to_scale] = scaler.fit_transform(transformed_df[features_to_scale])
                        st.success("Applied min-max scaling.")
                        
                    elif scaling_method == "Robust Scaling":
                        from sklearn.preprocessing import RobustScaler
                        scaler = RobustScaler()
                        transformed_df[features_to_scale] = scaler.fit_transform(transformed_df[features_to_scale])
                        st.success("Applied robust scaling.")
                        
                    elif scaling_method == "Log Transformation":
                        for col in features_to_scale:
                            # Handle non-positive values
                            min_val = transformed_df[col].min()
                            if min_val <= 0:
                                offset = abs(min_val) + 1
                                transformed_df[col] = np.log(transformed_df[col] + offset)
                                st.info(f"Added {offset} to {col} before log transformation to handle non-positive values.")
                            else:
                                transformed_df[col] = np.log(transformed_df[col])
                        st.success("Applied log transformation.")
                    
                    st.session_state.transformed_data = transformed_df
                except Exception as e:
                    st.error(f"Error applying scaling: {str(e)}")
        else:
            st.info("No numerical columns found for scaling.")
    
    with tabs[3]:
        st.subheader("Dimensionality Reduction")
        
        numerical_cols = transformed_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if len(numerical_cols) > 1:
            st.write("### Principal Component Analysis (PCA)")
            
            features_for_pca = st.multiselect("Select features for PCA", numerical_cols, key="pca_cols")
            
            if features_for_pca:
                n_components = st.slider("Number of components", 1, min(len(features_for_pca), 20), min(3, len(features_for_pca)))
                
                if st.button("Apply PCA"):
                    try:
                        # Apply PCA
                        pca = PCA(n_components=n_components)
                        pca_result = pca.fit_transform(transformed_df[features_for_pca])
                        
                        # Add PCA components to the dataframe
                        for i in range(n_components):
                            transformed_df[f'PCA_{i+1}'] = pca_result[:, i]
                        
                        # Display explained variance ratio
                        explained_variance = pca.explained_variance_ratio_
                        cumulative_variance = np.cumsum(explained_variance)
                        
                        st.write("Explained Variance Ratio:")
                        for i, var in enumerate(explained_variance):
                            st.write(f"PC{i+1}: {var:.4f} ({cumulative_variance[i]:.4f} cumulative)")
                        
                        # Plot explained variance
                        fig = px.line(
                            x=list(range(1, n_components+1)),
                            y=cumulative_variance,
                            labels={'x': 'Number of Components', 'y': 'Cumulative Explained Variance'},
                            title='Explained Variance by Components'
                        )
                        fig.update_traces(mode='lines+markers')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.success(f"Applied PCA and added {n_components} new columns.")
                        st.session_state.transformed_data = transformed_df
                    except Exception as e:
                        st.error(f"Error applying PCA: {str(e)}")
            
            # Feature selection options
            st.write("### Feature Selection")
            st.info("Feature selection options will be added in future updates.")
        else:
            st.info("Need at least two numerical columns for dimensionality reduction.")
    
    with tabs[4]:
        st.subheader("Preview Transformed Data")
        
        # Show the transformed data
        st.dataframe(transformed_df.head(10))
        
        # Compare original and transformed data
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Cleaned data shape:")
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        
        with col2:
            st.write("Transformed data shape:")
            st.write(f"Rows: {transformed_df.shape[0]}, Columns: {transformed_df.shape[1]}")
            
        # List of new features
        new_features = list(set(transformed_df.columns) - set(df.columns))
        if new_features:
            st.write("### New Features Created:")
            for feature in new_features:
                st.write(f"- {feature}")
        
        # Save transformed data
        if st.button("Finalize Transformed Data"):
            st.session_state.transformed_data = transformed_df
            st.success("Transformed data saved! You can now proceed to the Data Splitting step.")
        
        # Option to download transformed data
        st.markdown(get_download_link(transformed_df, "transformed_data.csv", "Download transformed data as CSV"), unsafe_allow_html=True)
    
    # Display tips
    with st.expander("Tips for Feature Engineering"):
        st.markdown("""
        - **Create domain-specific features** based on your understanding of the problem
        - **Transform variables** to achieve normality if needed for the algorithm
        - **Encode categorical variables** appropriately for your model
        - **Scale numerical features** for algorithms sensitive to feature scales (e.g., SVM, KNN)
        - **Reduce dimensionality** to avoid the curse of dimensionality
        - **Document your transformations** for reproducibility
        """) 