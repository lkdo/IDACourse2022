""" Mini library, implements linear least squares estimation. 
    Performs parameter estimation of functions of following form:
    y = [ f1(x), f2(x), ... , fp(x)] @ [param_1, param_2, ..., param_p]
    where y is a scalar, and x is a 1D array, fi are functions with scalar output 
    Also f(x) = [ f1(x), f2(x), ... , fp(x)]  """

import numpy as np
import matplotlib.pyplot as plt
from sys import exit
import inspect

def F(X, f):
  """ X = [x1, x2, ..., xn] vector of inputs; xi can be a vector itself
      f  = f(x); main model function  
      The input matrix 
      F = [ [ f1(x1), f2(x1), ..., fp(x1) ],
            [ f1(x2), f2(x2), ..., fp(x2) ], 
             ...
            [ f1(xn), f2(xn), ..., fp(xn) ] ], where
      f(x) = [ f1(x), f2(x), ..., fp(x) ]  """
  if (np.isscalar(X)):
      X = np.array([X])
  
  if ( (X.ndim>2) or (not callable(f)) or 
        (len(inspect.getfullargspec(f).args) != 1) ):    
    print(inspect.currentframe().f_code.co_name, "(): ", "ERROR Input parameters sanity check failed.")
    exit(-1)
  else:
    # Checking also that the output of f is scalar or 1D array 
    fx = f(X[0])
    if (np.isscalar(fx)):
      fx = np.array([fx])
    if (fx.ndim != 1):
      print(inspect.currentframe().f_code.co_name, "(): ", "ERROR Input parameters sanity check failed.")
      exit(-1)

  FX = []
  for x in X:
    FX.append(f(x))
  FX = np.array(FX)
  return FX
##########################################################

def run_linear_least_squares_estimation(X, Y, f):
  """ Inputs, Outputs, Input function (i.e. Model Structure) """
  if (np.isscalar(X)):
      X = np.array([X])
  if (np.isscalar(Y)):
      Y = np.array([Y])
  
  if ( (X.ndim > 2) or (Y.ndim != 1) or (X.shape[0] != Y.shape[0]) or 
          (not callable(f)) or (len(inspect.getfullargspec(f).args) != 1) ):    
    print(inspect.currentframe().f_code.co_name, "(): ", "ERROR Input parameters sanity check failed.")
    exit(-1)
  else:
    # Checking also that the output of f is scalar or 1D array 
    fx = f(X[0])
    if (np.isscalar(fx)):
      fx = np.array([fx])
    if (fx.ndim != 1):
      print(inspect.currentframe().f_code.co_name, "(): ", "ERROR Input parameters sanity check failed.")
      exit(-1)

  FX = F(X,f)
  P = np.linalg.inv(FX.transpose()@FX)
  params = P@FX.transpose()@Y
  return params, P
##########################################################

def run_linear_model(X, f, params):
  
  """ inputs, model functions, parameters """
  if (np.isscalar(X)):
      X = np.array([X])
  if (np.isscalar(params)):
      params = np.array([params])    
  
  if ( (X.ndim > 2) or (params.ndim !=1) or 
        (not callable(f)) or (len(inspect.getfullargspec(f).args) != 1) ):
    print(inspect.currentframe().f_code.co_name, "(): ", "ERROR Input parameters sanity check failed.")
    exit(-1)
  else:
    # Checking that the output of f is scalar or 1D array, and if it matches with param size 
    fx = f(X[0])
    if (np.isscalar(fx)):
      fx = np.array([fx])
    if ((fx.ndim != 1) or (fx.shape[0] != params.shape[0])):
      print(inspect.currentframe().f_code.co_name, "(): ", "ERROR Input parameters sanity check failed.")
      exit(-1)
  
  FX = F(X,f)
  Y = FX@params
  return Y
##########################################################

def plotgraph(data_k,data_Y,model_k,model_Y,title): # Make a simple graph
  if (np.isscalar(data_k)):
    data_k = np.array([data_k])
  if (np.isscalar(model_k)):
    model_k = np.array([model_k])
  if (np.isscalar(data_Y)):
    data_Y = np.array([data_Y])
  if (np.isscalar(model_Y)):
    model_Y = np.array([model_Y])

  if ( (data_k.ndim != 1) or (data_Y.ndim !=1) or (model_k.ndim != 1) or (model_Y.ndim != 1) or
        (data_k.shape[0] != data_Y.shape[0]) or (model_k.shape[0]!= model_Y.shape[0]) ):
    print(inspect.currentframe().f_code.co_name, "(): ", "ERROR Input parameters sanity check failed.")
    exit(-1)

  _, ax = plt.subplots(1, 1)
  ax.plot(data_k,data_Y,color='green', linestyle='none', marker='o',
     markerfacecolor='blue', markersize=8, label='Real Data')
  ax.plot(model_k,model_Y,color='orange', linestyle='dashed', marker='o',
     markerfacecolor='red', markersize=5, label="Model Generated Data")
  plt.title(title)
  plt.ylabel('Time [s]')
  plt.grid(True)
  plt.legend()
##########################################################


class RLLS:
  """ Recursive Linear Least Squares """

  def __init__(self, params, P, alpha = 1.0):
    if ( (params.ndim != 1) or (P.ndim != 2) or (np.isscalar(alpha) == False) or
        (alpha > 1) or (alpha < 0) or (params.shape[0] != P.shape[0]) ): 
      print(inspect.currentframe().f_code.co_name, "(): ", "ERROR Input parameters sanity check failed.")
      exit(-1)

    self.params = params
    self.P = P
    self.p = self.params.shape[0]
    self.alpha = alpha 
    """ alpha = forgetting factor ; 
        alpha = 1 for constant parameters, 
        alpha < 1 for time changing parameters """

  def update(self, fx, meas):
    if ( (fx.ndim != 1) or (fx.shape[0]!= self.params.shape[0]) or (np.isscalar(meas) == False)):
      print(inspect.currentframe().f_code.co_name, "(): ", "ERROR Input parameters sanity check failed.")
      return
    fx = np.array([fx]).transpose() # column vector now
    params = np.array([self.params]).transpose() # column vector now
    K = (self.P@fx)/(self.alpha + fx.transpose()@self.P@fx)
    y_pred = fx.transpose()@params
    self.params = ( params + K*(meas-y_pred) ).flatten()
    self.P = 1.0/self.alpha*(np.eye(self.p) - K@fx.transpose())@self.P
    
  def covariance_reset(self, k):
    if (k < 0):
      k = 1 
    self.P = k*np.eye(self.p)  