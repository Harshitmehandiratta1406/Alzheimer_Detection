# Alzheimer Detection System Using Machine Learning

This repository contains the implementation of an Alzheimer's detection system using machine learning techniques. The project leverages the OASIS dataset, which includes both cross-sectional and longitudinal MRI data, to develop and validate models capable of detecting Alzheimer's disease.

# Table of Contents

    ~ Objective
    ~ Installation
    ~ Datasets
    ~ Data Preprocessing
    ~ Model Training and Evaluation
    ~ Methodology
    ~ Limitations
    ~ Results
    ~ Usage
    ~ Bibliography
   



# Objective

The primary objective of this project is to develop a machine learning-based system for early detection of Alzheimer's disease using MRI data from the OASIS dataset. By applying various machine learning algorithms, the system aims to accurately classify patients with Alzheimer's and track the disease's progression over time.

# Installation

To get started, clone the repository and install the required dependencies:

 1. git clone https://github.com/Harshitmehandiratta1406/Alzheimer_Detection.git
 2. cd Alzheimer_Detection
 3. pip install -r requirements.txt



# Datasets

~ OASIS Cross-Sectional Dataset (Alzheimer_Detection\oasis_cross-sectional.csv): Contains MRI and clinical data collected at a single time point for each subject.
~ OASIS Longitudinal Dataset (Alzheimer_Detection\oasis_longitudinal.csv): Contains MRI and clinical data collected over multiple time points for each subject, allowing for the study of disease progression.

Download the dataset from (OASIS(https://sites.wustl.edu/oasisbrains/)).

# Data Preprocessing

The data preprocessing steps include:

   ~ Loading the datasets.
   ~ Inspecting the data for missing values and data types.
   ~ Cleaning the data by dropping unnecessary columns and rows with missing target values.
   ~ Imputing missing values using different strategies.
   ~ Encoding categorical variables.
   ~ Scaling features for model training.

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

   ~ Sampling Technique: Stratified sampling was used to ensure representative distribution of subjects across different stages of Alzheimer's  disease.
   ~ Sample Size: The sample size includes all available subjects from the OASIS datasets, with appropriate train-test splits to ensure robust model evaluation.
   ~ Data Collection Procedure: Data was sourced from the publicly available OASIS datasets, preprocessed, and used to extract relevant features for machine learning models.
   ~ Data Collection Instrument: MRI scans and clinical assessments were the primary instruments for data collection.

   # Machine Learning Process

    ~ Feature Selection and Extraction: Relevant features such as age, MRI-derived brain volumes, and cognitive scores were selected.
    ~ Model Training: Various algorithms including logistic regression, decision trees, random forests, support vector machines (SVM), and neural networks were trained and evaluated.
    ~ Parameter Tuning: Hyperparameters were optimized using techniques like grid search and cross-validation.

# Limitations

  1. Sample Size and Representativeness: While the OASIS dataset is comprehensive, the sample may not fully represent the broader population.
  2. Data Quality: Variability in MRI scan quality and clinical assessments could impact model accuracy.
  3. User Experience: The complexity of interpreting MRI data and model outputs can affect user adoption.
  4. Privacy and Security: Handling sensitive medical data requires stringent privacy and security measures.
  5. Generalizability: Models trained on OASIS data may not generalize well to other datasets or populations.
  6. Response Bias: Potential biases in clinical assessments and subject reporting.
  7. Subjective Nature of Qualitative Data: Variability in clinical assessments and cognitive tests.
  8. Cross-Sectional Design: Limited ability to infer causal relationships.
  9. Resource Constraints: Computational resources required for training and validating complex models.
  10. External Factors: Uncontrolled variables that could affect model performance.
  11. Strengths and Weaknesses: Detailed analysis of the model's performance, robustness, and areas needing improvement.

# Results

The best model parameters and accuracy scores are printed. The confusion matrix and classification report provide detailed performance metrics for the model.

# Usage

To use this project:

   1. Clone the repository.
   2. Install the required dependencies.
   3. Download the dataset and place it in the project directory.
   4. Run the Jupyter notebook or Python script to preprocess the data, train the model, and evaluate its performance.

# Biblography

 1. Marcus, D. S., et al. "Open Access Series of Imaging Studies (OASIS): Cross-Sectional MRI Data in Young, Middle-Aged, Nondemented, and Demented Older Adults." Journal of Cognitive Neuroscience, 2007.
 2. Marcus, D. S., et al. "Open Access Series of Imaging Studies: Longitudinal MRI Data in Nondemented and Demented Older Adults." Journal of Cognitive Neuroscience, 2010.
 3. Jack, C. R., et al. "The Alzheimer's Disease Neuroimaging Initiative (ADNI): MRI Methods." Journal of Magnetic Resonance Imaging, 2008.
 4. McKhann, G., et al. "The Diagnosis of Dementia Due to Alzheimer's Disease: Recommendations from the National Institute on Aging-Alzheimer's Association Workgroups on Diagnostic Guidelines for Alzheimer's Disease." Alzheimer's & Dementia, 2011.
