import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

def handle_imbalanced_data():
    st.header("7. Handle Imbalanced Data")
    
    # Check if we have split data
    if (st.session_state.X_train is None or 
        st.session_state.y_train is None or 
        st.session_state.feature_columns is None or 
        st.session_state.target_variable is None):
        st.warning("Please complete the Data Splitting step first.")
        return
    
    # Get data from session state
    X_train = st.session_state.X_train.copy()
    y_train = st.session_state.y_train.copy()
    feature_columns = st.session_state.feature_columns
    target_variable = st.session_state.target_variable
    
    st.markdown("""
    ### Check and Handle Class Imbalance
    
    Imbalanced datasets can lead to biased models that perform poorly on minority classes.
    """)
    
    # Check if target is categorical
    if not (y_train.dtype == 'object' or y_train.dtype.name == 'category' or y_train.nunique() < 10):
        st.info("This step is primarily for classification problems with categorical target variables. Your target appears to be continuous (regression problem).")
        
        # For regression, just show the distribution
        st.subheader("Target Variable Distribution")
        fig = px.histogram(y_train, title=f"Distribution of {target_variable}")
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("For regression problems, you can now proceed to the Evaluation Metrics step.")
        return
    
    # For classification problems, check class distribution
    class_counts = pd.DataFrame(y_train.value_counts())
    class_counts.columns = ["Count"]
    class_counts["Percentage"] = (class_counts["Count"] / len(y_train) * 100).round(2)
    
    st.subheader("Class Distribution")
    st.dataframe(class_counts)
    
    # Visualize class distribution
    fig = px.bar(
        class_counts.reset_index(), 
        x="index", 
        y="Count", 
        text="Percentage",
        title=f"Class Distribution of {target_variable}"
    )
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    
    # Check if imbalanced
    max_class_pct = class_counts["Percentage"].max()
    min_class_pct = class_counts["Percentage"].min()
    
    if max_class_pct / min_class_pct > 3:
        st.warning(f"Dataset appears imbalanced. The majority class is {max_class_pct:.2f}% while the minority class is only {min_class_pct:.2f}%.")
    else:
        st.success("Dataset appears relatively balanced.")
    
    # Resampling techniques
    st.subheader("Resampling Techniques")
    
    resampling_technique = st.selectbox(
        "Select resampling technique",
        [
            "None - Keep Original Data",
            "Random Oversampling (Duplicate Minority)",
            "SMOTE (Synthetic Minority Oversampling)",
            "Random Undersampling (Remove Majority)",
            "Combination (Under + Over Sampling)"
        ]
    )
    
    if resampling_technique != "None - Keep Original Data":
        if resampling_technique == "Random Oversampling (Duplicate Minority)":
            st.markdown("""
            **Random Oversampling** duplicates examples from the minority class to balance the class distribution.
            - Pros: Simple, no information loss
            - Cons: May cause overfitting due to duplicate samples
            """)
            
            sampling_strategy = st.slider(
                "Sampling Strategy (ratio of minority to majority class)", 
                0.1, 1.0, 1.0
            )
            
            if st.button("Apply Random Oversampling"):
                try:
                    # Apply oversampling
                    ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
                    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
                    
                    # Store in session state
                    st.session_state.X_train_original = X_train.copy()
                    st.session_state.y_train_original = y_train.copy()
                    st.session_state.X_train = X_resampled
                    st.session_state.y_train = y_resampled
                    
                    # Show new class distribution
                    new_class_counts = pd.DataFrame(y_resampled.value_counts())
                    new_class_counts.columns = ["Count"]
                    new_class_counts["Percentage"] = (new_class_counts["Count"] / len(y_resampled) * 100).round(2)
                    
                    st.success("Random oversampling applied successfully!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Original class distribution:")
                        st.dataframe(class_counts)
                    
                    with col2:
                        st.write("New class distribution:")
                        st.dataframe(new_class_counts)
                    
                    # Show counts before and after
                    original_counts = dict(Counter(y_train))
                    resampled_counts = dict(Counter(y_resampled))
                    
                    for cls in original_counts:
                        st.write(f"Class {cls}: {original_counts.get(cls, 0)} → {resampled_counts.get(cls, 0)} samples")
                    
                except Exception as e:
                    st.error(f"Error applying random oversampling: {str(e)}")
        
        elif resampling_technique == "SMOTE (Synthetic Minority Oversampling)":
            st.markdown("""
            **SMOTE** creates synthetic samples from the minority class instead of duplicating existing samples.
            - Pros: Avoids overfitting compared to random oversampling
            - Cons: May create unrealistic samples, sensitive to noise
            """)
            
            sampling_strategy = st.slider(
                "Sampling Strategy (ratio of minority to majority class)", 
                0.1, 1.0, 1.0
            )
            
            k_neighbors = st.slider(
                "Number of nearest neighbors to use", 
                1, 10, 5
            )
            
            # Check if data is suitable for SMOTE
            min_class_count = class_counts["Count"].min()
            if min_class_count < k_neighbors + 1:
                st.warning(f"The minority class has only {min_class_count} samples, which is less than k_neighbors+1 ({k_neighbors+1}). SMOTE may not work well.")
            
            if st.button("Apply SMOTE"):
                try:
                    # Apply SMOTE
                    smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=42)
                    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
                    
                    # Store in session state
                    st.session_state.X_train_original = X_train.copy()
                    st.session_state.y_train_original = y_train.copy()
                    st.session_state.X_train = X_resampled
                    st.session_state.y_train = y_resampled
                    
                    # Show new class distribution
                    new_class_counts = pd.DataFrame(y_resampled.value_counts())
                    new_class_counts.columns = ["Count"]
                    new_class_counts["Percentage"] = (new_class_counts["Count"] / len(y_resampled) * 100).round(2)
                    
                    st.success("SMOTE applied successfully!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Original class distribution:")
                        st.dataframe(class_counts)
                    
                    with col2:
                        st.write("New class distribution:")
                        st.dataframe(new_class_counts)
                    
                    # Show counts before and after
                    original_counts = dict(Counter(y_train))
                    resampled_counts = dict(Counter(y_resampled))
                    
                    for cls in original_counts:
                        st.write(f"Class {cls}: {original_counts.get(cls, 0)} → {resampled_counts.get(cls, 0)} samples")
                    
                except Exception as e:
                    st.error(f"Error applying SMOTE: {str(e)}")
        
        elif resampling_technique == "Random Undersampling (Remove Majority)":
            st.markdown("""
            **Random Undersampling** removes examples from the majority class to balance the class distribution.
            - Pros: Reduces computational cost, may avoid majority class bias
            - Cons: May lose important information from majority class
            """)
            
            sampling_strategy = st.slider(
                "Sampling Strategy (ratio of minority to majority class)", 
                0.1, 1.0, 1.0
            )
            
            if st.button("Apply Random Undersampling"):
                try:
                    # Apply undersampling
                    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
                    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
                    
                    # Store in session state
                    st.session_state.X_train_original = X_train.copy()
                    st.session_state.y_train_original = y_train.copy()
                    st.session_state.X_train = X_resampled
                    st.session_state.y_train = y_resampled
                    
                    # Show new class distribution
                    new_class_counts = pd.DataFrame(y_resampled.value_counts())
                    new_class_counts.columns = ["Count"]
                    new_class_counts["Percentage"] = (new_class_counts["Count"] / len(y_resampled) * 100).round(2)
                    
                    st.success("Random undersampling applied successfully!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Original class distribution:")
                        st.dataframe(class_counts)
                    
                    with col2:
                        st.write("New class distribution:")
                        st.dataframe(new_class_counts)
                    
                    # Show counts before and after
                    original_counts = dict(Counter(y_train))
                    resampled_counts = dict(Counter(y_resampled))
                    
                    for cls in original_counts:
                        st.write(f"Class {cls}: {original_counts.get(cls, 0)} → {resampled_counts.get(cls, 0)} samples")
                    
                except Exception as e:
                    st.error(f"Error applying random undersampling: {str(e)}")
        
        elif resampling_technique == "Combination (Under + Over Sampling)":
            st.markdown("""
            **Combination approach** uses both undersampling and oversampling to balance the class distribution.
            - Pros: Can be more effective than either technique alone
            - Cons: Parameter tuning can be more complex
            """)
            
            # Undersampling settings
            st.subheader("Undersampling Settings")
            undersampling_strategy = st.slider(
                "Undersampling Strategy (ratio of minority to majority class)", 
                0.1, 1.0, 0.5
            )
            
            # Oversampling settings
            st.subheader("Oversampling Settings")
            oversampling_method = st.selectbox(
                "Oversampling Method",
                ["Random Oversampling", "SMOTE"]
            )
            
            oversampling_strategy = st.slider(
                "Oversampling Strategy (ratio of minority to majority class after undersampling)", 
                0.1, 1.0, 1.0
            )
            
            if oversampling_method == "SMOTE":
                k_neighbors = st.slider(
                    "Number of nearest neighbors for SMOTE", 
                    1, 10, 5
                )
            
            if st.button("Apply Combination Approach"):
                try:
                    # First apply undersampling
                    rus = RandomUnderSampler(sampling_strategy=undersampling_strategy, random_state=42)
                    X_temp, y_temp = rus.fit_resample(X_train, y_train)
                    
                    # Then apply oversampling
                    if oversampling_method == "Random Oversampling":
                        oversampler = RandomOverSampler(sampling_strategy=oversampling_strategy, random_state=42)
                    else:  # SMOTE
                        oversampler = SMOTE(sampling_strategy=oversampling_strategy, k_neighbors=k_neighbors, random_state=42)
                    
                    X_resampled, y_resampled = oversampler.fit_resample(X_temp, y_temp)
                    
                    # Store in session state
                    st.session_state.X_train_original = X_train.copy()
                    st.session_state.y_train_original = y_train.copy()
                    st.session_state.X_train = X_resampled
                    st.session_state.y_train = y_resampled
                    
                    # Show new class distribution
                    new_class_counts = pd.DataFrame(y_resampled.value_counts())
                    new_class_counts.columns = ["Count"]
                    new_class_counts["Percentage"] = (new_class_counts["Count"] / len(y_resampled) * 100).round(2)
                    
                    st.success("Combination approach applied successfully!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Original class distribution:")
                        st.dataframe(class_counts)
                    
                    with col2:
                        st.write("New class distribution:")
                        st.dataframe(new_class_counts)
                    
                    # Show counts before and after
                    original_counts = dict(Counter(y_train))
                    resampled_counts = dict(Counter(y_resampled))
                    
                    for cls in original_counts:
                        st.write(f"Class {cls}: {original_counts.get(cls, 0)} → {resampled_counts.get(cls, 0)} samples")
                    
                except Exception as e:
                    st.error(f"Error applying combination approach: {str(e)}")
    
    # Alternative approaches
    st.subheader("Alternative Approaches")
    
    st.markdown("""
    Besides resampling, there are other approaches to handle imbalanced data:
    
    - **Adjust class weights**: Give higher weights to minority classes during model training
    - **Use appropriate evaluation metrics**: Accuracy can be misleading. Consider precision, recall, F1-score, or AUC-ROC
    - **Anomaly detection**: For extreme imbalance, treat it as an anomaly detection problem
    - **Ensemble methods**: Use ensemble techniques that are more robust to class imbalance
    - **Specialized algorithms**: Some algorithms handle imbalanced data better than others
    """)
    
    # Class weighting
    st.write("### Class Weighting Example")
    
    st.code('''
    # For most scikit-learn models:
    model = RandomForestClassifier(class_weight='balanced')
    
    # Or using custom weights:
    weights = {0: 1, 1: 10}  # Give class 1 ten times more weight
    model = RandomForestClassifier(class_weight=weights)
    ''')
    
    # Reset to original data
    if "X_train_original" in st.session_state and st.button("Reset to Original Data"):
        st.session_state.X_train = st.session_state.X_train_original.copy()
        st.session_state.y_train = st.session_state.y_train_original.copy()
        st.success("Reset to original training data!")
    
    # Display tips
    with st.expander("Tips for Handling Imbalanced Data"):
        st.markdown("""
        - **Choose resampling techniques** that match your data and problem
        - **Try different approaches** and compare their performance
        - **Be cautious with synthetic data** creation, which might introduce unrealistic patterns
        - **Focus on domain-specific performance metrics** rather than just balancing classes
        - **Consider cost-sensitive learning** if misclassification costs differ between classes
        """) 