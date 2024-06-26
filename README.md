# Alzheimer Detection System Using Machine Learning

This repository contains the implementation of an Alzheimer's detection system using machine learning techniques. The project leverages the OASIS dataset, which includes both cross-sectional and longitudinal MRI data, to develop and validate models capable of detecting Alzheimer's disease.

# Table of Contents

    ~ Objective
    ~ Installation
    ~ Datasets
    ~ Data Preprocessing
    ~ Model Training and Evaluation
    ~ Methodology
    ~ Results
    ~ Usage
   

# Objective

The primary objective of this project is to develop a machine learning-based system for early detection of Alzheimer's disease using MRI data from the OASIS dataset. By applying various machine learning algorithms, the system aims to accurately classify patients with Alzheimer's and track the disease's progression over time.

# Installation

To get started, clone the repository and install the required dependencies:

 1. git clone https://github.com/Harshitmehandiratta1406/Alzheimer_Detection.git
 2. cd Alzheimer_Detection
 3. pip install -r requirements.txt



# Datasets

1. OASIS Cross-Sectional Dataset (oasis_cross-sectional.csv): Contains MRI and clinical data collected at a single time point for each subject.
2. OASIS Longitudinal Dataset (oasis_longitudinal.csv): Contains MRI and clinical data collected over multiple time points for each subject, allowing for the study of disease progression.

Download the dataset from (OASIS(https://sites.wustl.edu/oasisbrains/)).

# Data Preprocessing

The data preprocessing steps include:

   1. Loading the datasets.
   2. Inspecting the data for missing values and data types.
   3. Cleaning the data by dropping unnecessary columns and rows with missing target values.
   4. Imputing missing values using different strategies.
   5. Encoding categorical variables.
   6. Scaling features for model training.

# Model Training and Evaluation

   # Hyperparameter Tuning

    Hyperparameter tuning is performed using RandomizedSearchCV to find the best parameters for the XGBoost classifier.
    
    parametros_gb = {
    "learning_rate": [0.01, 0.025, 0.005, 0.1, 0.15, 0.2, 0.3, 0.8, 0.9],
    "max_depth": [3, 5, 8, 10, 15, 20, 25, 30, 40, 50],
    "subsample": [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators": range(1, 100)
    }

    NUM_FOLDS = 5
    model_gb = XGBClassifier()
    gb_random = RandomizedSearchCV(estimator=model_gb, param_distributions=parametros_gb, n_iter=100, cv=NUM_FOLDS,
    verbose=0, random_state=42, n_jobs=-1, scoring='accuracy')
    gb_random.fit(X_train, y_train)

    print("Best parameters:", gb_random.best_params_)
    model_gb = gb_random.best_estimator_
    
   # Cross-Validation

   Cross-validation is used to evaluate the model's performance:

   cv_accuracy = cross_val_score(model_gb, x, y, cv=10, scoring='accuracy').mean()
   print("Cross-validation accuracy:", cv_accuracy)
   
   # Model Evaluation

   The model is evaluated using a confusion matrix and classification report:
   
   cm = confusion_matrix(y_test, y_pred)
   clr = classification_report(y_test, y_pred)
   
   plt.figure(figsize=(8, 8))
   sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=False, cmap='Blues')
   plt.xlabel("Predicted")
   plt.ylabel("Actual")
   plt.title("Confusion Matrix")
   plt.show()
   
   print("Classification Report:\n----------------------\n", clr)



# Methodology

  # Research Design

   1. Sampling Technique: Stratified sampling was used to ensure representative distribution of subjects across different stages of Alzheimer's  disease.
   2. Sample Size: The sample size includes all available subjects from the OASIS datasets, with appropriate train-test splits to ensure robust model evaluation.
   3. Data Collection Procedure: Data was sourced from the publicly available OASIS datasets, preprocessed, and used to extract relevant features for machine learning models.
   4. Data Collection Instrument: MRI scans and clinical assessments were the primary instruments for data collection.

   # Machine Learning Process

   1. Feature Selection and Extraction: Relevant features such as age, MRI-derived brain volumes, and cognitive scores were selected.
   2. Model Training: Various algorithms including logistic regression, decision trees, random forests, support vector machines (SVM), and neural networks were trained and evaluated.
   3. Parameter Tuning: Hyperparameters were optimized using techniques like grid search and cross-validation.



# Results

The best model parameters and accuracy scores are printed. The confusion matrix and classification report provide detailed performance metrics for the model.

# Usage

To use this project:

   1. Clone the repository.
   2. Install the required dependencies.
   3. Download the dataset and place it in the project directory.
   4. Run the Jupyter notebook or Python script to preprocess the data, train the model, and evaluate its performance.

