#Heart Disease Prediction Using Machine Learning
This project aims to predict the likelihood of heart disease based on a variety of medical and lifestyle factors. By applying different machine learning algorithms, the goal is to build an accurate model that assists healthcare professionals in identifying individuals at higher risk, thus enabling timely interventions.

Table of Contents
Features
Machine Learning Algorithms
Evaluation Metrics
Technologies Used
Project Objectives
Results
Future Work
How to Run
Features
The dataset used in this project consists of several key features related to heart disease risk factors:

Year (yr)
Cholesterol level (cholesterol)
Weight (weight)
Glucose level (gluc)
Diastolic blood pressure (ap_lo)
Systolic blood pressure (ap_hi)
Activity level (active)
Smoking habit (smoke)
These features serve as inputs to the machine learning models to predict whether an individual has heart disease.

Machine Learning Algorithms
The following algorithms have been used to train the model:

Logistic Regression
Random Forest Classifier
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
Decision Trees
Evaluation Metrics
The performance of each model is evaluated using the following metrics:

Accuracy: Measures the percentage of correct predictions.
Precision: Focuses on the accuracy of positive predictions.
Recall: Measures the ability of the model to capture all positive instances.
F1-Score: A balance between precision and recall.
ROC-AUC: Evaluates the model's ability to distinguish between classes.
Technologies Used
Python: The programming language used for model development.
Jupyter Notebook: For interactive code development.
Pandas and NumPy: For data manipulation and preprocessing.
Scikit-learn: For machine learning model development and evaluation.
Matplotlib and Seaborn: For data visualization and exploratory data analysis.
Project Objectives
Preprocessing: Handle missing values, normalize features, and prepare the dataset for training.
Feature Selection: Use techniques like SHAP (SHapley Additive exPlanations) to explore feature importance.
Model Development: Apply various machine learning algorithms to predict heart disease.
Model Comparison: Compare models using evaluation metrics and choose the best-performing one.
