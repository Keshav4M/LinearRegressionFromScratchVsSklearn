# Linear Regression: Custom Implementation and scikit-learn Comparison

## Overview

This repository provides an educational comparison of two approaches to linear regression: a custom implementation using gradient descent and a pre-built implementation using the `scikit-learn` library. The goal is to understand the fundamentals of linear regression and to see how both approaches work in practice.

## Repository Structure

- `data.csv`: The dataset used for linear regression. It contains two columns: the feature (X) and the target (y).
- `custom_linear_regression.ipynb`: Jupyter Notebook implementing linear regression from scratch using gradient descent.
- `sklearn_linear_regression.ipynb`: Jupyter Notebook implementing linear regression using the `scikit-learn` library.
- `comparison.ipynb`: Jupyter Notebook that compares the results of both approaches and visualizes the differences.

## Linear Regression Basics

Linear regression is a statistical method used to model the relationship between a dependent variable (target) and one or more independent variables (features). The model assumes a linear relationship between the variables. The goal is to find the best-fitting straight line (regression line) that predicts the target variable from the feature(s).

### The Linear Regression Model

The linear regression model can be expressed as:

\[ y = mx + b \]

Where:
- \( y \) is the target variable.
- \( x \) is the feature variable.
- \( m \) is the slope of the line.
- \( b \) is the y-intercept.

### Mean Squared Error (MSE)

The Mean Squared Error (MSE) is used to measure the accuracy of the linear regression model. It is calculated as the average of the squares of the errors (the difference between the actual and predicted values):

\[ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2 \]

Where:
- \( n \) is the number of data points.
- \( y_i \) is the actual value.
- \( \hat{y_i} \) is the predicted value.

## Custom Gradient Descent Implementation

Gradient descent is an optimization algorithm used to minimize the MSE by iteratively adjusting the parameters \( m \) and \( b \). The process involves calculating the gradient (partial derivatives) of the MSE with respect to each parameter and updating the parameters in the opposite direction of the gradient.

### Steps:
1. Initialize \( m \) and \( b \) to zero.
2. Compute the gradient of the MSE with respect to \( m \) and \( b \).
3. Update \( m \) and \( b \) using the gradients and a learning rate.
4. Repeat steps 2 and 3 for a specified number of iterations.

The `custom_linear_regression.ipynb` file provides a detailed implementation of this approach.

## scikit-learn Linear Regression

The `scikit-learn` library offers a simple and efficient tool for performing linear regression. It abstracts away the details of the optimization process, providing a straightforward API for fitting a linear model to the data.

### Steps:
1. Load the data.
2. Instantiate the `LinearRegression` model.
3. Fit the model to the data using the `fit` method.
4. Make predictions using the `predict` method.
5. Calculate the MSE using the `mean_squared_error` function.

The `sklearn_linear_regression.ipynb` file demonstrates this approach.

## Comparison

The `comparison.ipynb` file compares the results of both approaches by plotting the regression lines and the error values. This notebook helps to visualize the differences and understand the strengths and weaknesses of each method.

## Usage

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/linear-regression-comparison.git
   cd linear-regression-comparison
   ```

2. Ensure you have the necessary dependencies installed:
   ```sh
   pip install -r requirements.txt
   ```

3. Open the Jupyter Notebooks in your preferred environment (e.g., Jupyter Notebook, JupyterLab, VS Code):
   - `custom_linear_regression.ipynb` for the custom gradient descent implementation.
   - `sklearn_linear_regression.ipynb` for the scikit-learn implementation.
   - `comparison.ipynb` for comparing both approaches.

## Conclusion

This repository provides a hands-on approach to understanding linear regression by comparing a custom implementation using gradient descent with the pre-built implementation provided by `scikit-learn`. It serves as a valuable educational resource for learning about linear regression and the practical applications of these techniques.