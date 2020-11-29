import numpy as np


#######################################################
# DO NOT MODIFY ANY CODE OTHER THAN THOSE TODO BLOCKS #
#######################################################

def delta(N, D, X, y, w, b, loss="perceptron"):
    """
    Obtain the delta of w and b
    :param N: scalar, number of samples
    :param D: scalar, number of features
    :param X: (N, D) all training samples
    :param y: (N, ) all training labels
    :param w: (D, ), coefficient
    :param b: scalar, bias
    :param loss: the loss that will be applied
    :return: dw, db: the delta of w and b
    """
    dw, db = np.zeros(D), 0.0
    if loss == "perceptron":
        # Given the perceptron loss l(w,b) = 1/N sum( max( 0, -y(wx+b) ) )
        z = np.multiply(y, np.dot(X, w) + b)
        indicators = np.where(z <= 0.0)
        dw = -np.sum(np.multiply(X[indicators].T, y[indicators]), axis=1) / N
        db = -np.sum(y[indicators]) / N
        return dw, db
    else:
        # Given the logistic loss l(w, b) = sum( ln(1+exp( -y(wx+b) )) )
        z = np.multiply(y, np.dot(X, w) + b)
        dw = -np.dot(X.T, np.multiply(sigmoid(-z), y)) / N
        db = -np.sum(np.multiply(sigmoid(-z), y)) / N
        return dw, db


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data (either 0 or 1)
    - loss: loss type, either perceptron or logistic
	- w0: initial weight vector (a numpy array)
	- b0: initial bias term (a scalar)
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the final trained weight vector
    - b: scalar, the final trained bias term

    Find the optimal parameters w and b for inputs X and y.
    Use the *average* of the gradients for all training examples
    multiplied by the step_size to update parameters.	
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2

    w = np.zeros(D)
    if w0 is not None:
        w = w0

    b = 0
    if b0 is not None:
        b = b0

    # Transform to (-1, 1) labels
    y_transformed = np.zeros(N)
    for i in range(N):
        y_transformed[i] = -1.0 if y[i] == 0 else 1.0

    if loss == "perceptron":
        # TODO 1 : perform "max_iterations" steps of
        # gradient descent with step size "step_size"
        # to minimize perceptron loss
        for _ in range(max_iterations):
            dw, db = delta(N=N, D=D, X=X, y=y_transformed, w=w, b=b, loss=loss)
            w -= step_size * dw
            b -= step_size * db

    elif loss == "logistic":
        # TODO 2 : perform "max_iterations" steps of
        # gradient descent with step size "step_size"
        # to minimize logistic loss
        for _ in range(max_iterations):
            dw, db = delta(N=N, D=D, X=X, y=y_transformed, w=w, b=b, loss=loss)
            w -= step_size * dw
            b -= step_size * db

    else:
        raise NotImplementedError

    assert w.shape == (D,)
    return w, b


def sigmoid(z):
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after applying the sigmoid function 1/(1+exp(-z)).
    """
    # TODO 3 : fill in the sigmoid function
    value = 1.0 / (1.0 + np.exp(-z))
    return value


def binary_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    
    Returns:
    - preds: N-dimensional vector of binary predictions (either 0 or 1)
    """
    N, D = X.shape

    # TODO 4 : predict DETERMINISTICALLY (i.e. do not randomize)
    preds = np.dot(X, w) + b
    preds[preds > 0.0] = 1
    preds[preds <= 0.0] = 0

    assert preds.shape == (N,)
    return preds


def softmax(z):
    """
    Obtain the softmax of z
    :param z: (C, N) or (C, ), a numpy array
    :return:
    """
    z -= np.max(z, axis=0)
    value = np.exp(z) / np.sum(np.exp(z), axis=0)
    return value


def multiclass_train(X, y, C,
                     w0=None,
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5,
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data (0, 1, ..., C-1)
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform (stochastic) gradient descent

    Returns:
    - w: C-by-D weight matrix, where C is the number of classes and D 
    is the dimensionality of features.
    - b: a bias vector of length C, where C is the number of classes
	
    Implement multinomial logistic regression for multiclass 
    classification. Again for GD use the *average* of the gradients for all training 
    examples multiplied by the step_size to update parameters.
	
    You may find it useful to use a special (one-hot) representation of the labels, 
    where each label y_i is represented as a row of zeros with a single 1 in
    the column that corresponds to the class y_i. Also recall the tip on the 
    implementation of the softmax function to avoid numerical issues.
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0

    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)  # DO NOT CHANGE THE RANDOM SEED IN YOUR FINAL SUBMISSION

    # Transform y to one-hot representation
    y1hot = np.zeros((N, C))
    for i in range(N):
        y1hot[i][y[i]] = 1.0

    if gd_type == "sgd":

        for it in range(max_iterations):
            n = np.random.choice(N)
            # TODO 5 : perform "max_iterations" steps of
            # stochastic gradient descent with step size
            # "step_size" to minimize logistic loss. We already
            # pick the index of the random sample for you (n)
            x = X[n]
            P = softmax(np.dot(w, x.T) + b) - y1hot[n]
            dw, db = np.dot(P.reshape(C, 1), x.reshape(1, D)), P
            w -= step_size * dw
            b -= step_size * db

    elif gd_type == "gd":
        # TODO 6 : perform "max_iterations" steps of
        # gradient descent with step size "step_size"
        # to minimize logistic loss.
        for _ in range(max_iterations):
            P = softmax(np.dot(w, X.T) + b[:, None]) - y1hot.T
            dw, db = np.dot(P, X), np.sum(P, axis=1)
            w -= step_size * dw / N
            b -= step_size * db / N

    else:
        raise NotImplementedError

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained model, C-by-D 
    - b: bias terms of the trained model, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Predictions should be from {0, 1, ..., C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    # TODO 7 : predict DETERMINISTICALLY (i.e. do not randomize)

    y = np.dot(X, w.T) + b
    preds = np.argmax(y, axis=1)

    assert preds.shape == (N,)
    return preds
