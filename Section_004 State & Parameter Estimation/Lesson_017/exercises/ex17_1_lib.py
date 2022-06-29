import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

class DNLDS:
  """ Discrete-time Non-Linear State-Space Dynamic Model with noise:
            xk+1 = xk + f(xk,uk)*dt + Gamma*wk
            yk = h(xk) + vk
      with initial condition x0, and noise characterization E[wk] = 0, Cov(wk) = Q, E[vk] = 0, Cov(vk) = R """

  def __init__(self, f, h, G, Q, R, x0):
    
    # ToDo: size checks

    # Initialization     
    self.f = f # a function 
    self.h = h # a function 
    self.G = G
    self.Q = Q
    self.R = R
    self.x = x0
    self.rng = np.random.default_rng()

  def step(self,u,dt,simple=1):
    if (simple): # euler integration (first order hold)
        self.x = self.x + self.f(self.x, 0, u)*dt +  \
              dt*self.G@self.rng.multivariate_normal(np.zeros(self.Q.shape[0]),self.Q)
    else: # ode integration
        Y = odeint(self.f,self.x,np.array([0, dt]),args=(u,))
        self.x = Y[1]          
  
  def output(self):
    return self.h(self.x) + self.rng.multivariate_normal(np.zeros(self.R.shape[0]),self.R)
##########################################################

class EKF:
    """ Implementation of the discrete-time EKF """
      
    def __init__(self, f, h, dfdx, dhdx, G, Q, R, x0, P0, M=0, b=0):

        # ToDo: size checks
        self.f = f
        self.h = h
        self.dfdx = dfdx
        self.dhdx = dhdx
        self.G = G
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0
        self.n = self.x.shape[0]
        
        # Implements constraint M@x=b
        if (M==0):
            return 
        # ToDo: Implement size check
        self.M = M
        self.b = b

    def predict(self, u, dt, simple = 1):
        if (simple):  # euler integration (first order hold)
            self.x = self.x + self.f(self.x, 0, u)*dt
        else:
            Y = odeint(self.f,self.x,np.array([0, dt]),args=(u,))
            self.x = Y[1]  
        A = np.eye(self.x.shape[0]) + self.dfdx(self.x, u)*dt
        self.P = A@self.P@A.transpose() + dt*self.G@self.Q@self.G.transpose()

    def update(self, y, var = 0):
        H = self.dhdx(self.x)
        Pxy = self.P@H.transpose()
        Py = H@self.P@H.transpose()
        K = Pxy@np.linalg.inv(Py+self.R)
        y_est = self.h(self.x)
        self.x = self.x + K@(y-y_est)
        if (var == 0):
            # Simple Covariance Update
            self.P = self.P - K@(Py+self.R)@np.transpose(K)
        else:
            # Joseph Form Covariance Update
            IUK = np.eye(self.n) - K@H
            self.P = IUK@self.P@IUK.transpose()+K@self.R@K.transpose()

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

class SUT: 
    """ Scaled Unscented Transform """

    def __init__(self, alpha, beta, kappa, n):
        
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.n = n

        self.l = (self.alpha**2)*(self.n+self.kappa) - self.n # lambda parameter 

        # Init weights (constants) 
        self.W0m = self.l/(self.n + self.l)
        self.W0c = self.W0m + 1 - self.alpha**2 + self.beta
        self.Wi = 0.5/(self.n + self.l)
        """ sigma points weights """
    
    def create_points(self, x, P):
        sr_P = np.linalg.cholesky((self.n + self.l)*P)
        S = [x]
        for i in range(self.n):
            S.append(x+sr_P[:,i])
            S.append(x-sr_P[:,i])
        return S
##########################################################

#class CDT:       
#  """ Central Difference Transform """
##########################################################

class SPKF:
    """ Sigma Point Kalman Filter """

    def __init__(self,f,h,G,Q,R,x0,P0,spt,variant=0):

        self.f = f
        """ Continous state dynamics; dot(x) =  f(x,u) """

        self.h = h
        """ Measurement function y = h(x) """

        self.G = G
        """ Noise input matrix """

        self.x = x0
        """ Initial State """

        self.P = P0
        """ Initial Covariance """

        self.Q = Q
        """ Propagation noise matrix """

        self.R = R
        """ Measurement noise matrix """

        self.spt = spt
        """ Sigma point transform and parameters """

        self.n = x0.size
        """ State dimensionality """

        self.variant = variant # 0 - normal UKF, 1 - IUKF, 2 - UKFz

    def predict(self,u,dt,simple = 1):

        S = self.spt.create_points(self.x, self.P)
  
        # propagate the points 
        Sp = [ ]
        for i in range(len(S)):
            if (simple):
                # Euler faster
                Sp.append(S[i]+self.f(S[i],0,u)*dt)
            else:
                # ODE Int integration to make the state tranzition
                Y = odeint(self.f,S[i],np.array([0, dt]),args=(u,))
                Sp.append(Y[1]) # Y[0]=x(t=t0)
     
        # calculate the mean, covariance and cross-covariance of the set
        Xm = self.spt.W0m*Sp[0]
        for i in range(2*self.n):
            Xm = Xm + self.spt.Wi*Sp[i+1]

        Cx = self.spt.W0c*np.outer(Sp[0]-Xm,Sp[0]-Xm)+dt*self.Q
        for i in range(2*self.n):
            Cx = Cx + self.spt.Wi*np.outer(Sp[i+1]-Xm,Sp[i+1]-Xm)

        self.x = Xm
        self.P = Cx

    def update(self,meas,var=0):

        X = self.spt.create_points(self.x, self.P)
        
        Y = [ ]
        for i in range(2*self.n+1):
            Y.append(self.h(X[i]))

        # calculate the mean and covariance of the set
        if (self.variant == 2): # UKFz
            Ym = self.h(self.x)
        else:  
            Ym = self.spt.W0m*Y[0]
            for i in range(2*self.n):
                Ym = Ym + self.spt.Wi*Y[i+1]
        
        Cy = self.spt.W0c*np.outer(Y[0]-Ym,Y[0]-Ym)
        Cxy = self.spt.W0c*np.outer(X[0]-self.x,Y[0]-Ym)
        for i in range(2*self.n):
            Cy = Cy + self.spt.Wi*np.outer(Y[i+1]-Ym,Y[i+1]-Ym)
            Cxy = Cxy + self.spt.Wi*np.outer(X[i+1]-self.x,Y[i+1]-Ym)

        # Kalman Gain
        K = Cxy@np.linalg.inv(Cy + self.R)

        # Update mean and covarince
        if (self.variant == 1): # IUKF
            inn = meas - self.h(self.x)
        else:
            inn = meas - Ym

        self.x = self.x + K@inn
        
        if (var == 0):
            # Simple Covariance Update
            self.P = self.P - K@(Cy+self.R)@np.transpose(K)
        else:
            # -> Joseph Form Covariance Update
            # H = stochastic linearization derived from the fact that Cxy = PH' in the linear case [Skoglund, Gustafsson, Hendeby - 2019]
            self.H = Cxy.transpose()@np.linalg.inv(self.P)
            IKH =  np.eye(self.n) - K@self.H
            self.P = IKH@self.P@IKH.transpose()+K@self.R@K.transpose() # Joseph Form
##########################################################
