import numpy as np

from uzapy.math.activations import sigmoid


class Regression:
  def __init__(self, lr=0.0001, n_iters=1000) -> None:
    self.lr = lr
    self.n_iters = n_iters
    self.weights = None
    self.bias = None


class LinearRegression(Regression):
  def fit(self, X, y) -> None:
    # init parameters
    n_samples , n_features = X.shape
    self.weights = np.zeros(n_features)
    self.bias = 0

    for _ in range(self.n_iters):
      y_predicted = np.dot(X, self.weights) + self.bias

      dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
      db = (1/n_samples) * np.sum(y_predicted - y)

      # update parameters
      self.weights -= self.lr * dw
      self.bias -= self.lr * db

  def predict(self, X):
    return np.dot(X, self.weights) + self.bias


class LogisticRegression(Regression):
  def fit(self, X, y):
    # init parameters
    n_samples, n_features = X.shape
    self.weights = np.zeros(n_features)
    self.bias = 0

    # gradient descent
    for _ in range(self.n_iters):
      linear_model = np.dot(X, self.weights) + self.bias
      y_predicted = sigmoid(linear_model)

      dw = (1/n_samples) * np.dot(X.T, y_predicted - y)
      db = (1/n_samples) * np.sum(y_predicted - y)

      self.weights -= self.lr * dw
      self.bias -= self.lr * db

  def predict(self, X):
    linear_model = np.dot(X, self.weights) + self.bias
    y_predicted = sigmoid(linear_model)
    y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
    return y_predicted_cls 