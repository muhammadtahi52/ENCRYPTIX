# ENCRYPTIX 

# Iris Flower Classification

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

