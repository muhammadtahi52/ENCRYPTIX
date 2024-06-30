# ENCRYPTIX 

# Project 01: Iris Flower Classification

This project demonstrates the classification of iris flower species using a Support Vector Machine (SVM). The iris dataset is a classic dataset used in machine learning and statistics.

## Overview

This project uses the following steps:
1. Load and explore the Iris dataset.
2. Visualize the dataset using Seaborn.
3. Train a Support Vector Machine (SVM) model.
4. Evaluate the model's performance.
5. Provide a function to predict iris species based on user input.

## Installation

1. Clone the repository:
    sh
    git clone https://github.com/yourusername/iris-flower-classification.git
    cd iris-flower-classification
    

2. Install the required packages:
    sh
    pip install numpy pandas matplotlib seaborn scikit-learn
    

## Usage

To run the Iris Classifier script:

sh
python iris_classifier.py


You will be prompted to enter the values for sepal length, sepal width, petal length, and petal width. The script will then predict the iris species based on the input values.

## Model Evaluation

The model's performance is evaluated using the classification report and confusion matrix.

### Classification Report
The classification report provides metrics such as precision, recall, and F1-score for each class.

### Confusion Matrix
The confusion matrix shows the number of correct and incorrect predictions for each class.

## Prediction

To predict the iris species based on user input, use the predict_iris_class function:

python
def predict_iris_class(sepal_length, sepal_width, petal_length, petal_width):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    class_name = iris.target_names[prediction][0]
    return class_name


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

# Project 02: Sales Prediction (Advertising Impact Analysis)

## Overview
This project explores the impact of advertising channels (TV, Radio, Newspaper) on sales using machine learning techniques.

## Model Performance
- **Mean Squared Error (MSE)**: 2.908
- **R-squared Score (R²)**: 0.906

These metrics indicate strong predictive accuracy, with the model explaining approximately 90.6% of the variance in sales based on advertising spends.

## Analysis
- **Visualization**: Included scatter plot shows the alignment between actual and predicted sales values, demonstrating the model's effectiveness.
- **Conclusion**: The model performs well in predicting sales outcomes based on ad expenditures across different channels.

## Project 03: Credit Card Fraud Detection
This project involves the detection of fraudulent credit card transactions using machine learning algorithms. The dataset consists of transactions made by European cardholders in September 2013. It includes 492 frauds out of 284,807 transactions, making it highly unbalanced with the positive class (frauds) accounting for only 0.172% of all transactions.

## Overview
The main objective of this project is to build a model that can accurately identify fraudulent transactions. We have utilized two machine learning algorithms for this purpose:

## Random Forest Classifier
## Logistic Regression
Data Preparation
Dataset: The dataset used is creditcard.csv, which contains transaction data along with a label indicating whether the transaction is fraudulent.
Features: The features include various anonymized attributes (V1, V2, ..., V28) along with Amount and Time.
Target: The target variable is Class, where 0 indicates a normal transaction and 1 indicates a fraudulent transaction.
Exploratory Data Analysis
Class Distribution
The dataset is highly unbalanced. The distribution of classes is visualized below:


## Correlation Heatmap
A heatmap is used to visualize the correlations between the features. This helps in understanding the relationships and dependencies between different variables:


## Model Training and Evaluation
Random Forest Classifier
The Random Forest algorithm is used to train the model. The performance metrics are as follows:

Accuracy:
Error Rate:
Specificity:
Sensitivity:
Logistic Regression
Logistic Regression is also used to train the model. The performance metrics are as follows:

Accuracy:
Error Rate:
Specificity:
Sensitivity:

## Conclusion
This project demonstrates the use of machine learning algorithms to detect fraudulent credit card transactions. The Random Forest and Logistic Regression classifiers provide a comparative analysis, showing how different models can perform on highly imbalanced datasets.



