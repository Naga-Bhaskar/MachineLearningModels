import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

class MyLinearRegression:  
  theta = None

  def fit(self, X, y, option, alpha, epoch):
    X = np.concatenate((np.array(X), np.ones((X.shape[0], 1), dtype=np.float64)), axis=1)
    y = np.array(y)       
    if option.lower() in ['bgd', 'gd']:
      # Run batch gradient descent.
      self.theta = self.batchGradientDescent(X, y, alpha, epoch)      
    elif option.lower() in ['sgd']:
      # Run stochastic gradient descent.
      self.theta = self.stocGradientDescent(X, y, alpha, epoch)
    else:
      # Run solving the normal equation.      
      self.theta = self.normalEquation(X, y)
    
  def predict(self, X):
    X = np.concatenate((np.array(X), np.ones((X.shape[0], 1), dtype=np.float64)), axis=1)
    if isinstance(self.theta, np.ndarray):
      y_pred = np.dot(X, self.theta)
      return y_pred
    return None

  def batchGradientDescent(self, X, y, alpha=0.00001, epoch=100000):
    (m, n) = X.shape      
    theta = np.zeros((n, 1), dtype=np.float64)
    for iter in range(epoch):
      if (iter % 1000) == 0:
        print('- currently at %d epoch...' % iter)    
      for j in range(n):
        theta[j] = theta[j] - (alpha * (sum([(np.dot(X[i], theta) - y[i]) * X[i][j] for i in range(m)])[0]) / m)
    return theta

  def stocGradientDescent(self, X, y, alpha=0.000001, epoch=10000):
    (m, n) = X.shape
    theta = np.zeros((n, 1), dtype=np.float64)
    for iter in range(epoch):
      if (iter % 100) == 0:
        print('- currently at %d epoch...' % iter)
      for i in range(m):
        for j in range(n):
          theta[j] = theta[j] - alpha * (np.dot(X[i], theta) - y[i]) * X[i][j]
    return theta

  def normalEquation(self, X, y):
    theta = np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.dot(np.transpose(X), y))
    return theta

  @staticmethod
  def toyLinearRegression(df, feature_name, target_name, option, alpha, epoch):
    # This function performs a simple linear regression.
    # With a single feature (given by feature_name)
    # With a rescaling (for stability of test)
    x = rescaleVector(df[feature_name])
    y = rescaleVector(df[target_name])
    x_train = x.values.reshape(-1, 1)
    y_train = y.values.reshape(-1, 1)

    # Perform linear regression.    
    lr = MyLinearRegression()
    lr.fit(x_train, y_train, option, alpha, epoch)
    y_train_pred = lr.predict(x_train)
    
    # Return training error and (x_train, y_train, y_train_pred)
    return mean_squared_error(y_train, y_train_pred), (x_train, y_train, y_train_pred)



def getDataFrame(dataset):
  featureColumns = pd.DataFrame(dataset.data, columns=dataset.feature_names)
  targetColumn = pd.DataFrame(dataset.target, columns=['Target'])
  return featureColumns.join(targetColumn)

def rescaleVector(x):
    min = x.min()
    max = x.max()
    return pd.Series([(element - min)/(max - min) for element in x])

def splitTrainTest(df, size):
  X, y = df.drop('Target', axis=1), df.Target
  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=size, test_size=X.shape[0] - size, random_state=0)
  return (X_train, y_train), (X_test, y_test)

def toyLinearRegression(df, feature_name, target_name):
  # This function performs a simple linear regression.
  # With a single feature (given by feature_name)
  # With a rescaling (for stability of test)
  x = rescaleVector(df[feature_name])
  y = rescaleVector(df[target_name])
  x_train = x.values.reshape(-1, 1)
  y_train = y.values.reshape(-1, 1)

  # Perform linear regression.
  lr = LinearRegression()
  lr.fit(x_train, y_train)
  y_train_pred = lr.predict(x_train)
  
  # Return training error and (x_train, y_train, y_train_pred)
  return mean_squared_error(y_train, y_train_pred), (x_train, y_train, y_train_pred)


def testYourCode(df, feature_name, target_name, option, alpha, epoch):
  trainError0, (x_train0, y_train0, y_train_pred0) = toyLinearRegression(df, feature_name, target_name)
  trainError1, (x_train1, y_train1, y_train_pred1) = MyLinearRegression.toyLinearRegression(df, feature_name, target_name, option, alpha, epoch)
  return trainError0, trainError1

def main():
	HousingDataset = load_boston()
	DataFrame = getDataFrame(HousingDataset)
	Df = DataFrame[DataFrame.Target < 22.5328 + 2*9.1971]
	TrainError0, TrainError1 = testYourCode(Df, 'DIS', 'Target', option='sgd', alpha=0.001, epoch=500)
	print("Scikit's training error = %.6f / My training error = %.6f --> Difference = %.4f" % (TrainError0, TrainError1, np.abs(TrainError0 - TrainError1)))
	TrainError0, TrainError1 = testYourCode(Df, 'RM', 'Target', option='bgd', alpha=0.1, epoch=5000)
	print("Scikit's training error = %.6f / My training error = %.6f --> Difference = %.4f" % (TrainError0, TrainError1, np.abs(TrainError0 - TrainError1)))

if __name__ == '__main__':
	main()