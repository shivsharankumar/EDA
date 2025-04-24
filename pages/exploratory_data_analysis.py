import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from utils import plot_numerical_features, plot_categorical_features

def exploratory_data_analysis():
    st.header("3. Exploratory Data Analysis (EDA)")
    
    if st.session_state.data is None:
        st.warning("No data available for analysis. Please upload data in the Data Collection step.")
        return
    
    df = st.session_state.data
    
    st.markdown("""
    ### Understand Your Data
    
    Exploratory Data Analysis (EDA) helps you understand the patterns, relationships, and anomalies in your data.
    """)
    
    # Display tabs for different EDA aspects
    tabs = st.tabs(["Overview", "Distributions", "Relationships", "Custom Analysis"])
    
    with tabs[0]:
        st.subheader("Data Overview")
        
        # Basic statistics
        st.write("Basic Statistics")
        st.dataframe(df.describe().T)
        
        # Data types and missing values
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Data Types")
            dtypes = pd.DataFrame(df.dtypes, columns=['Data Type'])
            dtypes.index.name = 'Column'
            st.dataframe(dtypes)
        
        with col2:
            st.write("Missing Values")
            missing = pd.DataFrame(df.isna().sum(), columns=['Missing Values'])
            missing['Percentage'] = (df.isna().sum() / len(df) * 100).round(2)
            missing.index.name = 'Column'
            st.dataframe(missing)
        
        # Unique values for categorical features
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            st.write("Unique Values in Categorical Features")
            for col in categorical_cols:
                unique_count = df[col].nunique()
                if unique_count < 20:  # Only show if not too many unique values
                    st.write(f"{col}: {unique_count} unique values")
                    st.write(df[col].value_counts())
    
    with tabs[1]:
        st.subheader("Data Distributions")
        
        # Numerical features distributions
        plot_numerical_features(df)
        
        # Categorical features distributions
        plot_categorical_features(df)
    
    with tabs[2]:
        st.subheader("Relationships Between Features")
        
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(numerical_cols) >= 2:
            st.write("#### Scatter Plots")
            
            x_col = st.selectbox("X-axis", numerical_cols, key="scatter_x")
            y_col = st.selectbox("Y-axis", numerical_cols, key="scatter_y")
            
            color_col = None
            if categorical_cols:
                use_color = st.checkbox("Color by categorical feature")
                if use_color:
                    color_col = st.selectbox("Color by", categorical_cols)
            
            if color_col:
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{x_col} vs {y_col} by {color_col}")
            else:
                fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
            
            st.plotly_chart(fig, use_container_width=True)
        
        if categorical_cols and numerical_cols:
            st.write("#### Box Plots")
            
            cat_col = st.selectbox("Categorical Feature", categorical_cols, key="box_cat")
            num_col = st.selectbox("Numerical Feature", numerical_cols, key="box_num")
            
            if df[cat_col].nunique() <= 10:  # Only plot if not too many categories
                fig = px.box(df, x=cat_col, y=num_col, title=f"{num_col} by {cat_col}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Too many categories in {cat_col} to display a meaningful box plot.")
    
    with tabs[3]:
        st.subheader("Custom Analysis")
        
        st.markdown("""
        Create custom visualizations by selecting features and plot types.
        """)
        
        plot_type = st.selectbox("Select Plot Type", 
                                ["Histogram", "Box Plot", "Scatter Plot", "Bar Chart", 
                                 "Correlation Heatmap", "Pair Plot"])
        
        if plot_type == "Histogram":
            col = st.selectbox("Select Feature", df.columns)
            bins = st.slider("Number of Bins", 5, 100, 20)
            
            fig = px.histogram(df, x=col, nbins=bins, title=f"Histogram of {col}")
            st.plotly_chart(fig, use_container_width=True)
            
        elif plot_type == "Box Plot":
            y_col = st.selectbox("Select Feature for Y-axis", df.columns)
            
            fig = px.box(df, y=y_col, title=f"Box Plot of {y_col}")
            st.plotly_chart(fig, use_container_width=True)
            
        elif plot_type == "Scatter Plot":
            x_col = st.selectbox("Select Feature for X-axis", df.columns)
            y_col = st.selectbox("Select Feature for Y-axis", df.columns, index=1 if len(df.columns) > 1 else 0)
            
            fig = px.scatter(df, x=x_col, y=y_col, title=f"Scatter Plot: {x_col} vs {y_col}")
            st.plotly_chart(fig, use_container_width=True)
            
        elif plot_type == "Bar Chart":
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if categorical_cols:
                col = st.selectbox("Select Categorical Feature", categorical_cols)
                
                if df[col].nunique() <= 20:  # Only show if not too many unique values
                    value_counts = df[col].value_counts().reset_index()
                    value_counts.columns = [col, 'Count']
                    
                    fig = px.bar(value_counts, x=col, y='Count', title=f"Count of {col}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Too many unique values in {col} to display a meaningful bar chart.")
            else:
                st.info("No categorical features found for bar chart.")
                
        elif plot_type == "Correlation Heatmap":
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if len(numerical_cols) > 1:
                selected_cols = st.multiselect("Select Features", numerical_cols, default=numerical_cols[:8])
                
                if selected_cols:
                    corr = df[selected_cols].corr()
                    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Correlation Heatmap")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough numerical features for correlation analysis.")
                
        elif plot_type == "Pair Plot":
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if len(numerical_cols) > 1:
                selected_cols = st.multiselect("Select Features (2-5 recommended)", numerical_cols, default=numerical_cols[:3])
                
                if len(selected_cols) >= 2:
                    if len(selected_cols) <= 5:  # Limit to avoid overloading
                        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                        hue_col = None
                        
                        if categorical_cols:
                            use_hue = st.checkbox("Color by categorical feature")
                            if use_hue:
                                hue_col = st.selectbox("Color by", categorical_cols)
                        
                        st.write("Generating pair plot...")
                        
                        # Create a Seaborn pair plot
                        fig, ax = plt.subplots(figsize=(10, 8))
                        pair_plot = sns.pairplot(df[selected_cols] if not hue_col else df[selected_cols + [hue_col]], 
                                             hue=hue_col,
                                             diag_kind='kde')
                        
                        st.pyplot(pair_plot.fig)
                    else:
                        st.warning("Please select 5 or fewer features for the pair plot.")
            else:
                st.info("Not enough numerical features for pair plot.")
    
    # Save EDA insights
    st.subheader("Document Your EDA Insights")
    eda_insights = st.text_area(
        "Key Insights from EDA", 
        placeholder="What patterns or relationships did you observe? Any outliers or anomalies? What features look promising?",
        key="eda_insights"
    )
    
    if st.button("Save EDA Insights"):
        st.session_state.eda_insights = eda_insights
        st.success("EDA insights saved! You can now proceed to the Data Cleaning step.")
        
    # Display tips
    with st.expander("Tips for Effective EDA"):
        st.markdown("""
        - **Look for patterns** and relationships between features
        - **Identify outliers** that might affect your analysis
        - **Understand distributions** of your features
        - **Check for imbalances** in your target variable
        - **Document your insights** to inform your modeling strategy
        """) 