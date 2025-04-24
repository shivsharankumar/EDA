import streamlit as st

def business_understanding():
    st.header("1. Business Understanding")
    
    st.markdown("""
    ### Define Your Objectives
    
    Before diving into data analysis, it's crucial to understand what you're trying to achieve.
    
    #### Key Questions to Answer:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ##### Problem Type:
        - **Classification:** Predicting categories (e.g., spam/not spam)
        - **Regression:** Predicting continuous values (e.g., house prices)
        - **Clustering:** Finding natural groupings in data
        - **Recommendation:** Suggesting items to users
        - **Time Series:** Forecasting future values
        """)
    
    with col2:
        st.markdown("""
        ##### Success Metrics:
        - **Classification:** Accuracy, Precision, Recall, F1-Score, AUC-ROC
        - **Regression:** MSE, RMSE, MAE, R-squared
        - **Clustering:** Silhouette Score, Davies-Bouldin Index
        - **Business KPIs:** Revenue impact, cost reduction, etc.
        """)
    
    st.subheader("Define Your Project")
    
    # Problem statement input
    st.text_area("Problem Statement", 
                placeholder="Describe the problem you're trying to solve...",
                key="problem_statement")
    
    # Project objectives
    st.text_area("Project Objectives", 
                placeholder="List your main objectives...",
                key="project_objectives")
    
    # Project type
    problem_type = st.selectbox("Problem Type", 
                             ["Classification", "Regression", "Clustering", 
                              "Recommendation", "Time Series", "Other"])
    
    # Success metrics based on problem type
    if problem_type == "Classification":
        metrics = st.multiselect("Success Metrics", 
                              ["Accuracy", "Precision", "Recall", "F1-Score", 
                               "AUC-ROC", "Log Loss", "Business KPI"])
    elif problem_type == "Regression":
        metrics = st.multiselect("Success Metrics", 
                              ["Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)", 
                               "Mean Absolute Error (MAE)", "R-squared", "Business KPI"])
    elif problem_type == "Clustering":
        metrics = st.multiselect("Success Metrics", 
                              ["Silhouette Score", "Davies-Bouldin Index", 
                               "Calinski-Harabasz Index", "Business KPI"])
    else:
        metrics = st.text_input("Success Metrics", 
                             "Specify your success metrics...")
    
    st.subheader("Stakeholders")
    stakeholders = st.text_area("Who will use the model output?", 
                            placeholder="List the stakeholders and how they'll use the model...",
                            key="stakeholders")
    
    st.subheader("Key Constraints")
    constraints = st.text_area("Are there any constraints to consider?", 
                           placeholder="E.g., interpretability requirements, legal considerations, deployment constraints...",
                           key="constraints")
    
    # Save button
    if st.button("Save Business Understanding"):
        st.session_state.business_details = {
            "problem_statement": st.session_state.get("problem_statement", ""),
            "project_objectives": st.session_state.get("project_objectives", ""),
            "problem_type": problem_type,
            "metrics": metrics,
            "stakeholders": st.session_state.get("stakeholders", ""),
            "constraints": st.session_state.get("constraints", "")
        }
        st.success("Business details saved! Navigate to the next step: Data Collection.")
        
    # Display tips
    with st.expander("Tips for Business Understanding"):
        st.markdown("""
        - **Be specific** about your objectives and success criteria
        - **Involve stakeholders** early in the process
        - **Understand the business impact** of your model's predictions
        - **Consider ethical implications** of your analysis
        - **Align with business goals** rather than just technical metrics
        """) 