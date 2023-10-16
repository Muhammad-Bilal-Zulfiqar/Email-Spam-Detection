# Email Spam Detection

This project focuses on classifying emails as either spam or non-spam using various machine learning algorithms. The implemented code utilizes popular Python libraries like NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn to preprocess data, train multiple models, and evaluate their performances.

## Project Overview

The primary objectives of this project are as follows:
- Data preprocessing involving handling missing values and feature extraction using the CountVectorizer
- Implementing and evaluating different machine learning models such as Naive Bayes, Logistic Regression, and Support Vector Classifier (with both linear and rbf kernels)
- Assessing the model performance through accuracy scores, confusion matrix, and classification reports
- Determining if the model is overfitting or not using K-Fold cross-validation

## Data Preprocessing

The code involves the following preprocessing steps:
- Dropping irrelevant columns and handling missing values
- Feature extraction using the CountVectorizer to convert text data into a numerical format suitable for machine learning algorithms

## Model Evaluation

The project evaluates the performance of the following machine learning algorithms:
- Multinomial Naive Bayes
- Logistic Regression
- Support Vector Classifier (with both linear and rbf kernels)

The evaluation includes computing accuracy scores, generating a confusion matrix for visual representation, and providing a comprehensive classification report.

## Results and Model Saving

The project showcases the accuracy of each model and analyzes whether the model is overfitting or not using K-Fold cross-validation. The final trained Logistic Regression model is saved as 'Email Spam Detection.pkl' using the joblib library.

## Usage

Users can utilize this code to build and evaluate their email spam detection models. They can modify the code for their datasets and experiment with different machine learning algorithms to improve classification accuracy.

## Requirements

The code utilizes several Python libraries, including NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn. Ensure these libraries are installed before running the code.

## Contribution

Contributions to this project are welcome. Feel free to open issues and submit pull requests for any improvements or additional features.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

