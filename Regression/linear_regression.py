import numpy as np
import pandas as pd


############################################################################
# DO NOT MODIFY CODES ABOVE 
# DO NOT CHANGE THE INPUT AND OUTPUT FORMAT
############################################################################

# Part 1.1
def mean_square_error(w, X, y):
    """
    Compute the mean square error of a model parameter w on a test set X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test features
    - y: A numpy array of shape (num_samples, ) containing test labels
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    n, y_predicted = len(y), np.dot(X, w)
    err = np.sum(np.power(y - y_predicted, 2), dtype=np.float) / n
    return err


# Part 1.2
def linear_regression_noreg(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing features
    - y: A numpy array of shape (num_samples, ) containing labels
    Returns:
    - w: a numpy array of shape (D, )
    """
    # w* = (X^T X)^(-1) X^T y
    w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
    return w


# Part 1.3
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing features
    - y: A numpy array of shape (num_samples, ) containing labels
    - lambd: a float number specifying the regularization parameter
    Returns:
    - w: a numpy array of shape (D, )
    """
    D = np.shape(X)[1]
    I = np.eye(D, k=0)
    w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)+lambd*I), X.T), y)
    return w


# Part 1.4
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training features
    - ytrain: A numpy array of shape (num_training_samples, ) containing training labels
    - Xval: A numpy array of shape (num_val_samples, D) containing validation features
    - yval: A numpy array of shape (num_val_samples, ) containing validation labels
    Returns:
    - bestlambda: the best lambda you find among 2^{-14}, 2^{-13}, ..., 2^{-1}, 1.
    """
    lambds, best_loss, best_lambda = [pow(2.0, -x) for x in range(15)], float('inf'), None
    for lambd in lambds:
        w = regularized_linear_regression(X=Xtrain, y=ytrain, lambd=lambd)
        mse = mean_square_error(w=w, X=Xval, y=yval)
        if mse < best_loss:
            best_lambda = lambd
            best_loss = mse

    return best_lambda


# Part 1.5
def mapping_data(X, P):
    """
    Augment the data to [X, X^2, ..., X^p]
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training features
    - P: An integer that indicates the degree of the polynomial regression
    Returns:
    - X: The augmented dataset. You might find np.insert useful.
    """
    raw = X
    for p in range(2, P+1):
        x_poly = np.power(raw, p)
        X = np.column_stack((X, x_poly))

    return X


"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""
