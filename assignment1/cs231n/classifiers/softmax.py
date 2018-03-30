import numpy as np
from random import shuffle
#from past.builtins import xrange

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
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X, W)
  #subtract max value in order to avoid overflow
  scores = scores - np.max(scores, axis=1)[:, np.newaxis]
  for i in range(num_train):
    cum_score = np.sum(np.exp(scores[i]))
    class_score = np.exp(scores[i,y[i]])
    loss += -np.log(class_score/cum_score)
    
    dW[:,y[i]] -= ((cum_score-class_score)/cum_score)*X[i]
    for j in range(num_classes):
        if j==y[i]:
            continue
        dW[:,j] += (np.exp(scores[i,j])/cum_score)*X[i]
        
  loss /= num_train
  dW /= num_train
  loss += reg*np.sum(W*W)
  dW += 2*reg*W
    
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
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X, W)
  #subtract max value in order to avoid overflow
  scores = scores - np.max(scores, axis=1)[:, np.newaxis]
  exp_scores = np.exp(scores)
  softmax_scores = exp_scores/(np.sum(exp_scores, axis=1)[:, np.newaxis])

  loss = -np.sum(np.log(softmax_scores[range(num_train), y]))
  loss /= num_train
  loss += reg*np.sum(W*W)


  mat = softmax_scores
  cum_exp_minus_correct_score = np.sum(exp_scores, axis=1) - exp_scores[range(num_train), y]
  mat[range(num_train), y] = -1*cum_exp_minus_correct_score/np.sum(exp_scores, axis=1)
  dW = np.dot(X.T, mat)
  dW /= num_train
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

