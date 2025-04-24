import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from imblearn.over_sampling import SMOTE
import os
import io
import base64

# Import page modules
from pages.business_understanding import business_understanding
from pages.data_collection import data_collection
from pages.exploratory_data_analysis import exploratory_data_analysis
from pages.data_cleaning import data_cleaning
from pages.feature_engineering import feature_engineering
from pages.data_splitting import data_splitting
from pages.handle_imbalanced_data import handle_imbalanced_data
from pages.evaluation_metrics import evaluation_metrics
from pages.environment_setup import environment_setup

# Set page configuration
st.set_page_config(page_title="Data Analysis Pipeline", layout="wide")

def main():
    st.title("Data Analysis Pipeline")
    st.markdown("""
    This app guides you through a complete data analysis process:
    1. **Business Understanding** - Define your objectives and success metrics
    2. **Data Collection** - Upload and explore your dataset
    3. **Exploratory Data Analysis (EDA)** - Visualize and understand your data
    4. **Data Cleaning** - Handle missing values, duplicates, and outliers
    5. **Feature Engineering** - Transform and create new features
    6. **Data Splitting** - Prepare data for modeling
    7. **Handle Imbalanced Data** - Apply techniques for imbalanced datasets
    8. **Evaluation Metrics** - Understand how to evaluate your model
    9. **Environment Setup** - Best practices for your ML environment
    """)
    
    # Create a sidebar for navigation
    st.sidebar.title("Navigation")
    pages = [
        "Business Understanding",
        "Data Collection",
        "Exploratory Data Analysis",
        "Data Cleaning",
        "Feature Engineering",
        "Data Splitting",
        "Handle Imbalanced Data",
        "Evaluation Metrics",
        "Environment Setup"
    ]
    
    # Initialize session state variables
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'cleaned_data' not in st.session_state:
        st.session_state.cleaned_data = None
    if 'transformed_data' not in st.session_state:
        st.session_state.transformed_data = None
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'X_val' not in st.session_state:
        st.session_state.X_val = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'y_val' not in st.session_state:
        st.session_state.y_val = None
    if 'feature_columns' not in st.session_state:
        st.session_state.feature_columns = None
    if 'target_variable' not in st.session_state:
        st.session_state.target_variable = None
    if 'file_name' not in st.session_state:
        st.session_state.file_name = None
    if 'business_details' not in st.session_state:
        st.session_state.business_details = None
    if 'data_source_info' not in st.session_state:
        st.session_state.data_source_info = None
    if 'eda_insights' not in st.session_state:
        st.session_state.eda_insights = None
    if 'X_train_original' not in st.session_state:
        st.session_state.X_train_original = None
    if 'y_train_original' not in st.session_state:
        st.session_state.y_train_original = None
    
    # Page selection
    selection = st.sidebar.radio("Go to", pages)
    
    # Display the selected page
    if selection == "Business Understanding":
        business_understanding()
    elif selection == "Data Collection":
        data_collection()
    elif selection == "Exploratory Data Analysis":
        exploratory_data_analysis()
    elif selection == "Data Cleaning":
        data_cleaning()
    elif selection == "Feature Engineering":
        feature_engineering()
    elif selection == "Data Splitting":
        data_splitting()
    elif selection == "Handle Imbalanced Data":
        handle_imbalanced_data()
    elif selection == "Evaluation Metrics":
        evaluation_metrics()
    elif selection == "Environment Setup":
        environment_setup()

if __name__ == "__main__":
    main() 