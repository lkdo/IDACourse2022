import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

def sizeCheck(A,B,G,H,Q,R,x0):
    if ((A.ndim != 2) or (B.ndim != 2) or (G.ndim != 2) or
            (H.ndim != 2) or (Q.ndim != 2) or (R.ndim != 2) or 
            (x0.ndim != 1) or 
            (A.shape[0] != A.shape[1]) or
            (Q.shape[0] != Q.shape[1]) or
            (R.shape[0] != R.shape[1]) or 
            (A.shape[0] != x0.shape[0]) or
            (B.shape[0] != A.shape[0]) or
            (G.shape[0] != A.shape[0]) or
            (G.shape[1] != Q.shape[0]) or
            (H.shape[0] != A.shape[0]) or
            (H.shape[1] != R.shape[0])):
        print("Incorect initializaton for dynamical system. Exiting.")
        return False
    return True
##########################################################

class DLDS:
  """ Discrete-time State-Space Model with noise:
            xk+1 = A*xk + B*uk + Gamma*wk
            yk = H*xk + vk
      with initial condition x0, and noise characterization E[wk] = 0, Cov(wk) = Q, E[vk] = 0, Cov(vk) = R """

  def __init__(self, A, B, G, H, Q, R, x0):
    # basic checks
    if ( not sizeCheck(A, B, G, H, Q, R, x0) ):
      exit(-1)

    # Initialization     
    self.A = A
    self.B = B
    self.G = G
    self.H = H
    self.Q = Q
    self.R = R
    self.x = x0
    self.rng = np.random.default_rng()

  def step(self,u):
    if ((u.ndim != 1) or (u.shape[0]!=self.B.shape[1])):
      print("Incorect dimensions for input signal. Skipping step.")
      return
    self.x = self.A@self.x + self.B@u + \
              self.G@self.rng.multivariate_normal(np.zeros(self.Q.shape[0]),self.Q)
  
  def output(self):
    return self.H@self.x + self.rng.multivariate_normal(np.zeros(self.R.shape[0]),self.R)

  def is_observable(self):
    # Computer the observability matrix
    for i in range(self.x.shape[0]):
        o = self.H@np.linalg.matrix_power(self.A,i)
        if "O" in locals():
            O = np.concatenate((O,o))
        else:
            O = o
    if (self.x.shape[0] == np.linalg.matrix_rank(O)):
        return True
    else:
        return False
##########################################################

class KalmanFilter:
    """ Implementation of the discrete-time Kalman Filter for linear systems """
      
    def __init__(self, A, B, G, H, Q, R, x0, P0, M=0, b=0):

        # basic checks
        if ( not sizeCheck(A, B, G, H, Q, R, x0) ):
            exit(-1)
        # extra check 
        if ((P0.ndim !=2) or 
                (P0.shape[0]!=P0.shape[1]) or
                (P0.shape[0]!=x0.shape[0]) ) :
            print("Incorect initializaton for dynamical system. Exiting.")
            exit(-1)
         
        self.A = A
        self.B = B
        self.G = G
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0
        
        # Implements equality constraint M@x=b
        if (np.isscalar(M)):
            return 
        # ToDo: Implement size check
        self.M = M
        self.b = b

    def predict(self, u):
        self.x = self.A@self.x +self.B@u
        self.P = self.A@self.P@self.A.transpose() + self.G@self.Q@self.G.transpose()

    def update(self, y, var = 0):
        Pxy =  self.P@self.H.transpose()
        Py = self.H@self.P@self.H.transpose()
        K = Pxy@np.linalg.inv(Py+self.R)
        y_est = self.H@self.x
        self.x = self.x + K@(y-y_est)
        if ( var == 0):
            # Simple Covariance Update
            self.P = self.P - K@(Py+self.R)@K.transpose()
        else:
            # Joseph form for Covariance update
            IUK = np.eye(self.x.shape[0])-K@self.H
            self.P = IUK@self.P@IUK.transpose()+K@self.R@K.transpose() # Joseph Form

    def apply_eq_constraint(self):
        #W = np.linalg.inv(self.P)
        #Wi = np.linalg.inv(W)

        # Option 1
        #Wi = self.P
        
        # Option 2
        Wi = np.eye(self.x.shape[0])

        A = np.linalg.inv(self.M@Wi@self.M.transpose())
        Lambda = Wi@self.M.transpose()*A
        
        self.x -= Lambda@(self.M@self.x-self.b)
        self.P = (np.eye(self.x.shape[0]) - Lambda@self.M)@self.P
##########################################################