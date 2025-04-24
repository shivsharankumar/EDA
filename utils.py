import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import io
import base64

def load_data(file):
    """Load data from uploaded file (CSV or Excel)"""
    file_extension = file.name.split('.')[-1]
    
    if file_extension.lower() == 'csv':
        df = pd.read_csv(file)
    elif file_extension.lower() in ['xlsx', 'xls']:
        df = pd.read_excel(file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None
        
    return df

def get_download_link(df, filename="data.csv", text="Download data as CSV"):
    """Generate a download link for a dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def show_data_info(df):
    """Display basic information about the dataframe"""
    st.subheader("Data Overview")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Rows: {df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")
    
    with col2:
        st.write(f"Duplicate rows: {df.duplicated().sum()}")
        st.write(f"Missing values: {df.isna().sum().sum()}")
    
    st.subheader("Data Sample")
    st.dataframe(df.head())
    
    st.subheader("Data Types")
    dtypes = pd.DataFrame(df.dtypes, columns=['Data Type'])
    dtypes.index.name = 'Column'
    st.dataframe(dtypes)
    
    st.subheader("Missing Values")
    missing = pd.DataFrame(df.isna().sum(), columns=['Missing Values'])
    missing['Percentage'] = (df.isna().sum() / len(df) * 100).round(2)
    missing.index.name = 'Column'
    st.dataframe(missing)

def plot_numerical_features(df):
    """Create visualizations for numerical features"""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if not numerical_cols:
        st.info("No numerical features found in the dataset.")
        return
    
    st.subheader("Numerical Features Distribution")
    
    # Distribution plots
    for col in numerical_cols:
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Add boxplot
        fig = px.box(df, y=col, title=f"Boxplot of {col}")
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    if len(numerical_cols) > 1:
        st.subheader("Correlation Matrix")
        corr = df[numerical_cols].corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                        title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)

def plot_categorical_features(df):
    """Create visualizations for categorical features"""
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_cols:
        st.info("No categorical features found in the dataset.")
        return
    
    st.subheader("Categorical Features Distribution")
    
    for col in categorical_cols:
        # Count plot
        value_counts = df[col].value_counts().reset_index()
        value_counts.columns = [col, 'Count']
        
        if len(value_counts) > 10:
            st.write(f"⚠️ {col} has {len(value_counts)} unique values. Showing top 10.")
            value_counts = value_counts.head(10)
        
        fig = px.bar(value_counts, x=col, y='Count', title=f"Count of {col}")
        st.plotly_chart(fig, use_container_width=True)

def detect_outliers(df, method='iqr'):
    """Detect outliers in numerical features"""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if not numerical_cols:
        return {}
    
    outliers = {}
    
    for col in numerical_cols:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
            outlier_count = len(outlier_indices)
            
            if outlier_count > 0:
                outlier_percentage = (outlier_count / len(df)) * 100
                outliers[col] = {
                    'count': outlier_count,
                    'percentage': outlier_percentage,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
    
    return outliers 