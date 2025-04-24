import streamlit as st

def environment_setup():
    st.header("9. Environment Setup")
    
    st.markdown("""
    ### Best Practices for Machine Learning Environment
    
    Setting up a proper environment for your machine learning projects is crucial for reproducibility, collaboration, and efficient workflows.
    """)
    
    # Display tabs for different aspects of environment setup
    tabs = st.tabs([
        "Version Control", 
        "Virtual Environments", 
        "Experiment Tracking", 
        "Reproducibility",
        "Project Structure"
    ])
    
    with tabs[0]:
        st.subheader("Version Control")
        
        st.markdown("""
        ### Git for ML Projects
        
        Version control is essential for tracking changes to your code and collaborating with others.
        
        #### Key Git Commands:
        ```bash
        # Initialize a repository
        git init
        
        # Stage changes
        git add .
        
        # Commit changes
        git commit -m "Descriptive message about changes"
        
        # Create and switch to a branch
        git checkout -b new-feature
        
        # Push to remote repository
        git push origin main
        ```
        
        #### Best Practices:
        - Use clear, descriptive commit messages
        - Create branches for new features/experiments
        - Use `.gitignore` to exclude large data files and sensitive information
        - Consider Git LFS for large model files
        
        #### What to Version Control:
        - Code and documentation
        - Configuration files
        - Small sample datasets
        - Model metadata and performance metrics
        
        #### What NOT to Version Control:
        - Large datasets
        - Model checkpoints and large binaries
        - Credentials and sensitive information
        - Environment-specific files
        """)
    
    with tabs[1]:
        st.subheader("Virtual Environments")
        
        st.markdown("""
        ### Isolating Dependencies
        
        Virtual environments allow you to create isolated environments for different projects.
        
        #### Using venv (Python):
        ```bash
        # Create virtual environment
        python -m venv myenv
        
        # Activate on Windows
        myenv\\Scripts\\activate
        
        # Activate on Unix/macOS
        source myenv/bin/activate
        
        # Install dependencies
        pip install -r requirements.txt
        
        # Freeze dependencies
        pip freeze > requirements.txt
        ```
        
        #### Using conda:
        ```bash
        # Create conda environment
        conda create --name myenv python=3.9
        
        # Activate environment
        conda activate myenv
        
        # Install packages
        conda install numpy pandas scikit-learn
        
        # Save environment
        conda env export > environment.yml
        
        # Create from file
        conda env create -f environment.yml
        ```
        
        #### Docker for Reproducible Environments:
        ```dockerfile
        FROM python:3.9-slim
        
        WORKDIR /app
        
        COPY requirements.txt .
        RUN pip install --no-cache-dir -r requirements.txt
        
        COPY . .
        
        CMD ["python", "main.py"]
        ```
        """)
    
    with tabs[2]:
        st.subheader("Experiment Tracking")
        
        st.markdown("""
        ### Tracking Your ML Experiments
        
        Experiment tracking helps you organize and compare different runs of your models.
        
        #### Tools for Experiment Tracking:
        
        **MLflow:**
        ```python
        import mlflow
        
        # Start a run
        mlflow.start_run()
        
        # Log parameters
        mlflow.log_param("learning_rate", 0.01)
        mlflow.log_param("batch_size", 32)
        
        # Log metrics
        mlflow.log_metric("accuracy", 0.85)
        mlflow.log_metric("f1_score", 0.82)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # End run
        mlflow.end_run()
        ```
        
        **Weights & Biases:**
        ```python
        import wandb
        
        # Initialize run
        wandb.init(project="my-project")
        
        # Config
        wandb.config.learning_rate = 0.01
        wandb.config.batch_size = 32
        
        # Log metrics
        wandb.log({"accuracy": 0.85, "f1_score": 0.82})
        
        # Finish run
        wandb.finish()
        ```
        
        #### What to Track:
        - Hyperparameters
        - Metrics
        - Model architecture/parameters
        - Environment details
        - Dataset information
        - Visualizations
        """)
    
    with tabs[3]:
        st.subheader("Reproducibility")
        
        st.markdown("""
        ### Ensuring Reproducible Results
        
        Reproducibility is crucial for scientific integrity and collaboration.
        
        #### Key Practices:
        
        **Set Random Seeds:**
        ```python
        import numpy as np
        import torch
        import random
        import tensorflow as tf
        
        def set_seeds(seed=42):
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            tf.random.set_seed(seed)
        
        set_seeds()
        ```
        
        **Data Version Control:**
        - Use DVC (Data Version Control) for large datasets
        - Document data preprocessing steps
        - Save data splits with random seeds
        
        **Dependency Management:**
        - Pin dependency versions in requirements.txt
        - Use Docker containers for complete environment isolation
        - Document hardware specifications
        
        **Pipeline Configuration:**
        - Use configuration files for experiment settings
        - Separate config from code
        - Document the entire pipeline
        """)
    
    with tabs[4]:
        st.subheader("Project Structure")
        
        st.markdown("""
        ### Organizing ML Projects
        
        A well-organized project structure makes it easier to understand, maintain, and share your work.
        
        #### Example Project Structure:
        ```
        my-ml-project/
        ├── data/
        │   ├── raw/              # Original, immutable data
        │   ├── processed/        # Cleaned, transformed data
        │   └── external/         # Data from external sources
        ├── notebooks/            # Exploratory Jupyter notebooks
        ├── src/                  # Source code
        │   ├── __init__.py
        │   ├── data/             # Data processing scripts
        │   ├── features/         # Feature engineering scripts
        │   ├── models/           # Model definition and training
        │   └── visualization/    # Visualization scripts
        ├── tests/                # Unit tests
        ├── configs/              # Configuration files
        ├── models/               # Saved models
        ├── outputs/              # Generated outputs (figures, etc.)
        ├── docs/                 # Documentation
        ├── .gitignore            # Specifies intentionally untracked files
        ├── environment.yml       # Conda environment file
        ├── requirements.txt      # Pip dependencies
        └── README.md             # Project description and setup guide
        ```
        
        #### Best Practices:
        - Separate data, code, and outputs
        - Use relative paths for portability
        - Include documentation
        - Maintain a clear README with setup instructions
        - Add a LICENSE file
        - Create a .gitignore file for your project
        """)
    
    # Final summary
    st.subheader("Environment Checklist")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ✅ **Version Control**
        - [ ] Initialize Git repository
        - [ ] Create .gitignore file
        - [ ] Use branches for features/experiments
        
        ✅ **Virtual Environment**
        - [ ] Create an isolated environment
        - [ ] Document dependencies
        - [ ] Consider containerization
        """)
    
    with col2:
        st.markdown("""
        ✅ **Experiment Tracking**
        - [ ] Choose a tracking tool
        - [ ] Track parameters and metrics
        - [ ] Log artifacts and models
        
        ✅ **Reproducibility**
        - [ ] Set random seeds
        - [ ] Version datasets
        - [ ] Document hardware and software versions
        """)
    
    # Display tips
    with st.expander("Tips for Environment Setup"):
        st.markdown("""
        - **Automate repetitive tasks** with scripts or Makefiles
        - **Document installation steps** for new team members
        - **Use pre-commit hooks** to enforce code quality
        - **Create environment setup scripts** for quick project onboarding
        - **Consider using a pipeline orchestration tool** for complex workflows
        """) 