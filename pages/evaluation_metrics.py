import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    mean_squared_error, mean_absolute_error, r2_score
)

def evaluation_metrics():
    st.header("8. Evaluation Metrics")
    
    # Check if required session state variables exist
    if not all(key in st.session_state for key in ['feature_columns', 'target_variable']):
        st.warning("Please complete the Data Splitting step first to define feature columns and target variable.")
        return
    
    # Check if we have split data
    if (st.session_state.feature_columns is None or 
        st.session_state.target_variable is None):
        st.warning("Please complete the Data Splitting step first.")
        return
    
    # Get target variable from session state
    target_variable = st.session_state.target_variable
    
    st.markdown("""
    ### Select Appropriate Evaluation Metrics
    
    Choosing the right metrics is crucial for evaluating your model's performance.
    """)
    
    # Check problem type
    if st.session_state.y_train is None:
        st.warning("Please complete the Data Splitting step first.")
        return
    
    y_train = st.session_state.y_train.copy()
    
    is_classification = False
    if y_train.dtype == 'object' or y_train.dtype.name == 'category' or y_train.nunique() < 10:
        is_classification = True
    
    if is_classification:
        problem_type = st.radio(
            "Problem Type",
            ["Binary Classification", "Multiclass Classification"],
            disabled=True,
            index=0 if y_train.nunique() == 2 else 1
        )
        
        st.subheader("Classification Metrics")
        
        metrics_tab1, metrics_tab2, metrics_tab3 = st.tabs(["Basic Metrics", "Confusion Matrix", "ROC and PR Curves"])
        
        with metrics_tab1:
            st.markdown("""
            ### Basic Classification Metrics
            
            - **Accuracy**: Proportion of correct predictions. Use when classes are balanced.
            - **Precision**: Proportion of positive identifications that were actually correct. Use when false positives are costly.
            - **Recall**: Proportion of actual positives that were identified correctly. Use when false negatives are costly.
            - **F1 Score**: Harmonic mean of precision and recall. Use for balanced evaluation of precision and recall.
            """)
            
            # Example metrics calculation
            st.write("#### Example Metrics Calculation")
            
            if problem_type == "Binary Classification":
                # Example metric code
                st.code('''
                # Binary classification metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                # Calculate metrics
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                
                # Average precision and recall for multiclass using different averaging strategies
                precision_macro = precision_score(y_true, y_pred, average='macro')
                recall_weighted = recall_score(y_true, y_pred, average='weighted')
                ''')
            else:
                # Example metric code for multiclass
                st.code('''
                # Multiclass classification metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                # Calculate metrics
                accuracy = accuracy_score(y_true, y_pred)
                
                # For multiclass, need to specify averaging strategy
                precision_macro = precision_score(y_true, y_pred, average='macro')
                precision_weighted = precision_score(y_true, y_pred, average='weighted')
                
                recall_macro = recall_score(y_true, y_pred, average='macro')
                recall_weighted = recall_score(y_true, y_pred, average='weighted')
                
                f1_macro = f1_score(y_true, y_pred, average='macro')
                f1_weighted = f1_score(y_true, y_pred, average='weighted')
                ''')
            
            # Choosing the right metric
            st.write("#### When to Use Each Metric")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Use Accuracy When:**")
                st.markdown("""
                - Classes are balanced
                - False positives and false negatives are equally important
                - Simple, intuitive metric is needed
                """)
                
                st.write("**Use Precision When:**")
                st.markdown("""
                - False positives are costly
                - Example: Spam detection (don't want to mark legitimate emails as spam)
                """)
            
            with col2:
                st.write("**Use Recall When:**")
                st.markdown("""
                - False negatives are costly
                - Example: Disease detection (don't want to miss positive cases)
                """)
                
                st.write("**Use F1 Score When:**")
                st.markdown("""
                - Need balance between precision and recall
                - Classes are imbalanced
                """)
        
        with metrics_tab2:
            st.markdown("""
            ### Confusion Matrix
            
            A confusion matrix provides a detailed breakdown of correct and incorrect classifications for each class.
            """)
            
            # Confusion matrix explanation
            st.write("#### Components of a Confusion Matrix (Binary Classification)")
            
            confusion_matrix_df = pd.DataFrame(
                [["True Negative (TN)", "False Positive (FP)"],
                 ["False Negative (FN)", "True Positive (TP)"]],
                index=["Actual Negative", "Actual Positive"],
                columns=["Predicted Negative", "Predicted Positive"]
            )
            
            st.table(confusion_matrix_df)
            
            # Calculations from confusion matrix
            st.write("#### Metrics Derived from Confusion Matrix")
            
            metrics_df = pd.DataFrame(
                [
                    ["Accuracy", "(TP + TN) / (TP + TN + FP + FN)", "Overall correctness"],
                    ["Precision", "TP / (TP + FP)", "Exactness"],
                    ["Recall", "TP / (TP + FN)", "Completeness"],
                    ["Specificity", "TN / (TN + FP)", "True negative rate"],
                    ["F1 Score", "2 * (Precision * Recall) / (Precision + Recall)", "Harmonic mean of precision and recall"]
                ],
                columns=["Metric", "Formula", "Interpretation"]
            )
            
            st.table(metrics_df)
            
            # Example code
            st.write("#### Code to Generate Confusion Matrix")
            
            st.code('''
            from sklearn.metrics import confusion_matrix
            import seaborn as sns
            import matplotlib.pyplot as plt
            
            # Generate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                      xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.show()
            ''')
        
        with metrics_tab3:
            st.markdown("""
            ### ROC and Precision-Recall Curves
            
            - **ROC (Receiver Operating Characteristic) Curve**: Plot of True Positive Rate vs. False Positive Rate.
            - **AUC (Area Under Curve)**: Aggregate measure of performance across all classification thresholds.
            - **Precision-Recall Curve**: Plot of Precision vs. Recall at different thresholds.
            """)
            
            st.write("#### When to Use Each Curve")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Use ROC Curve When:**")
                st.markdown("""
                - Classes are approximately balanced
                - Want to visualize tradeoff between TPR and FPR
                - Want to compare models using AUC
                """)
            
            with col2:
                st.write("**Use Precision-Recall Curve When:**")
                st.markdown("""
                - Classes are imbalanced
                - Focus is on positive class (minority class)
                - False negatives and false positives have different costs
                """)
            
            # Example code
            st.write("#### Code to Generate ROC and PR Curves")
            
            st.code('''
            from sklearn.metrics import roc_curve, auc, precision_recall_curve
            import matplotlib.pyplot as plt
            
            # Get prediction probabilities for the positive class
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            
            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = auc(recall, precision)
            
            plt.subplot(1, 2, 2)
            plt.plot(recall, precision, color='blue', lw=2, label=f'AUC = {pr_auc:.2f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            ''')
    
    else:  # Regression
        st.subheader("Regression Metrics")
        
        metrics_tab1, metrics_tab2 = st.tabs(["Common Metrics", "Visualization"])
        
        with metrics_tab1:
            st.markdown("""
            ### Common Regression Metrics
            
            - **Mean Squared Error (MSE)**: Average squared difference between predicted and actual values. Sensitive to outliers.
            - **Root Mean Squared Error (RMSE)**: Square root of MSE. Same unit as the target variable.
            - **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual values. Less sensitive to outliers.
            - **R² (Coefficient of Determination)**: Proportion of variance in the target variable explained by the model. Ranges from 0 to 1 (or can be negative if model is worse than baseline).
            - **Adjusted R²**: R² adjusted for the number of predictors.
            """)
            
            # Example metrics calculation
            st.write("#### Example Metrics Calculation")
            
            st.code('''
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            import numpy as np
            
            # Calculate metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # Calculate adjusted R²
            n = len(y_true)  # number of samples
            p = X_test.shape[1]  # number of features
            adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            ''')
            
            # Choosing the right metric
            st.write("#### When to Use Each Metric")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Use MSE/RMSE When:**")
                st.markdown("""
                - Larger errors should be penalized more
                - The target variable doesn't have outliers
                - Scale of errors is important
                """)
                
                st.write("**Use MAE When:**")
                st.markdown("""
                - Less emphasis on outliers is needed
                - Interested in average magnitude of errors
                - Want a metric robust to outliers
                """)
            
            with col2:
                st.write("**Use R² When:**")
                st.markdown("""
                - Need to know how much variance is explained
                - Want a scale-free metric (between 0 and 1)
                - Comparing different models
                """)
                
                st.write("**Use Adjusted R² When:**")
                st.markdown("""
                - Comparing models with different numbers of features
                - Want to penalize complex models
                """)
        
        with metrics_tab2:
            st.markdown("""
            ### Visualizing Regression Performance
            
            Visualizations help understand model performance beyond single metrics.
            """)
            
            # Example visualizations
            st.write("#### Common Regression Visualizations")
            
            # Example code
            st.write("#### Code for Regression Visualization")
            
            st.code('''
            import matplotlib.pyplot as plt
            import numpy as np
            
            # 1. Actual vs Predicted plot
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            
            # Add perfect prediction line
            min_val = min(min(y_test), min(y_pred))
            max_val = max(max(y_test), max(y_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Actual vs Predicted Values')
            
            # 2. Residual plot
            residuals = y_test - y_pred
            
            plt.figure(figsize=(10, 6))
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residual Plot')
            
            # 3. Residual histogram
            plt.figure(figsize=(10, 6))
            plt.hist(residuals, bins=20, alpha=0.7, color='blue')
            
            plt.xlabel('Residual Value')
            plt.ylabel('Frequency')
            plt.title('Histogram of Residuals')
            
            plt.tight_layout()
            plt.show()
            ''')
    
    # Cross-validation for both classification and regression
    st.subheader("Validation Strategies")
    
    st.markdown("""
    ### Cross-Validation
    
    Cross-validation helps estimate model performance more reliably by testing on multiple train-test splits.
    
    - **K-Fold Cross-Validation**: Split data into k folds, train on k-1 folds and test on the remaining fold, repeat k times.
    - **Stratified K-Fold**: Ensures each fold has the same proportion of classes (for classification).
    - **Time Series Cross-Validation**: For time series data, respects temporal order.
    """)
    
    # Example code
    st.write("#### Code for Cross-Validation")
    
    if is_classification:
        st.code('''
        from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
        from sklearn.metrics import make_scorer, f1_score
        
        # Define model
        model = RandomForestClassifier()
        
        # 1. Basic K-fold CV
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
        
        # 2. Stratified K-fold CV (for classification)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted')
        
        # 3. Custom scoring function
        f1_scorer = make_scorer(f1_score, average='weighted')
        cv_scores = cross_val_score(model, X, y, cv=5, scoring=f1_scorer)
        
        print(f"CV Scores: {cv_scores}")
        print(f"Mean CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        ''')
    else:
        st.code('''
        from sklearn.model_selection import cross_val_score, KFold
        from sklearn.metrics import make_scorer, mean_squared_error
        import numpy as np
        
        # Define model
        model = RandomForestRegressor()
        
        # 1. Basic K-fold CV
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # 2. Custom scoring function (negative because sklearn maximizes scores)
        rmse_scorer = make_scorer(lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)), 
                                greater_is_better=False)
        
        cv_scores = cross_val_score(model, X, y, cv=kf, scoring=rmse_scorer)
        
        # Convert back to positive for reporting
        cv_scores_positive = -cv_scores
        
        print(f"RMSE CV Scores: {cv_scores_positive}")
        print(f"Mean RMSE: {cv_scores_positive.mean():.4f} ± {cv_scores_positive.std():.4f}")
        ''')
    
    # Display tips
    with st.expander("Tips for Evaluation Metrics"):
        st.markdown("""
        - **Choose metrics aligned with business goals**, not just technical considerations
        - **Use multiple metrics** to get a comprehensive view of model performance
        - **Consider class imbalance** when selecting metrics for classification
        - **Report confidence intervals** or standard deviations with your metrics
        - **Use cross-validation** to get more reliable performance estimates
        """) 