import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    N, D = X.shape
    C = W.shape[1]
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    for i in range(N):
        z = X[i] @ W
        a = np.exp(z)
        asum = np.sum(a)
        loss += np.log(asum) - z[y[i]]
        dW[:, y[i]] -= X[i]
        dW += np.repeat(X[i], C).reshape(D, C) * a / asum
    loss /= N
    dW /= N
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    N, D = X.shape

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    Z = X @ W
    Z = Z - Z.max(axis=1, keepdims=True)
    A = np.exp(Z)
    cs = A.sum(axis=1)
    log_cs = np.log(cs)
    loss = log_cs - Z[np.arange(N), y]
    loss = loss.mean()
    loss += reg * np.sum(W * W)

    dZ = np.zeros_like(Z)
    dZ[np.arange(N), y] = -1/N
    dZ += A * ((1/N)/cs).reshape(-1, 1)
    dW = X.T @ dZ
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
