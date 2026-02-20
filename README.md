# Foundations of Machine Learning: Optimization, Classification, and Regression

**Author:** Ian Chen  
**Institution:** University of Southern California (USC) - Electrical and Computer Engineering (ECE)

## Overview
This repository serves as a practical exploration of core machine learning paradigms. Through a series of hands-on Python notebooks, this project demonstrates the mathematical foundations and programmatic implementation of predictive models. The focus is on building intuition for how learning algorithms optimize over data, transitioning from from-scratch mathematical implementations to leveraging industry-standard frameworks like PyTorch and Scikit-Learn.

## Key Concepts & Skills Acquired

Through the development of these projects, I deepened my understanding of several critical machine learning domains:

* **Optimization & Loss Functions:** Implemented Gradient Descent from scratch, gaining a mathematical understanding of how weights and biases are updated to minimize Mean Squared Error (MSE) and Binary Cross-Entropy loss.
* **Neural Network Architecture:** Designed and trained Fully-Connected Networks (FCNs) using PyTorch, managing the end-to-end pipeline from tensor manipulation to calculating forward passes and backpropagation.
* **Model Validation & Generalization:** Explored the theoretical and practical aspects of the bias-variance trade-off. Implemented k-fold cross-validation to rigorously evaluate model performance and utilized techniques to mitigate overfitting and underfitting.
* **Classical ML & Ensemble Methods:** Gained practical experience applying diverse algorithms, including Logistic Regression, Ridge/Lasso Regression, Naive Bayes, Random Forests, and Gradient Boosting, to various datasets (such as Iris and California Housing).
* **Data Visualization:** Utilized Matplotlib to visually interpret model behavior, including plotting regression lines, training loss curves over iterations, and complex decision boundaries for classification tasks.

## Project Breakdown

### 1. Gradient Descent & Linear Optimization (`1_Gradient_Descent.ipynb`)
This notebook explores the mechanics of learning through gradient descent. 
* Formulates linear regression mathematically.
* Features custom implementations to calculate predictions, residuals, gradients, and cost.
* Compares custom-built optimization loops with PyTorch's automated tensor operations.

### 2. Neural Network Classification & Model Validation (`2_Classification.ipynb`)
This section shifts focus to classification tasks and the critical evaluation of model robustness.
* Builds a PyTorch-based Fully-Connected Network evaluated on the Iris dataset.
* Implements robust data splitting and three-fold cross-validation methodologies.
* Investigates the theoretical differences in training dynamics between probabilistic models (like Naive Bayes) and gradient-based neural networks.

### 3. Comprehensive Applied Machine Learning (`3_Additional_ML_Topics.ipynb`)
A broader application of machine learning pipelines utilizing `pandas` and `scikit-learn`.
* Preprocesses and manipulates data structures using Pandas.
* Trains and evaluates a wide array of linear and ensemble models.
* Visualizes the effectiveness of the sigmoid function in logistic regression and plots comparative decision boundaries.

## Technologies Used
* **Languages:** Python 3
* **Deep Learning:** PyTorch
* **Machine Learning:** Scikit-Learn
* **Data Science:** NumPy, Pandas
* **Visualization:** Matplotlib