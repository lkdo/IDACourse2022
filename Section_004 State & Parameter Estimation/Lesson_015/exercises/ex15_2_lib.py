import numpy as np
import matplotlib.pyplot as plt
from sys import exit
import inspect

class ARX:
    """ A generic simulator for an ARX (autoregressive exogenous) SISO dynamic model:
    Yk-1 = [yk-1, ..., yk-pa]  i.e. a na slot memory for previous outputs
    Uk = [uk, ..., uk-pb+1]  i.e. a nb slot memory for current + previous inputs
    f(Yk-1,Uk) = [ f_1(Yk-1,Uk), ... , f_p(Yk-1,Uk) ] i.e. the model function
    yk = f(Yk-1,Uk)@[param_1, param_2, ..., param_p] i.e. the output equation  """
 
    def __init__(self, Y0, U0, params, f):
        self.Y = Y0 # previous na outputs 
        self.pa = Y0.size 
        self.U = U0 # previous nb inputs 
        self.pb = U0.size
        self.params = params # ARX model parameters 
        self.p = params.size
        self.f = f # this is the ARX model function 

    def output(self):
        f = self.f(np.concatenate([self.Y,self.U]))
        if (f.shape[0] != self.p):
            print(inspect.currentframe().f_code.co_name," ERROR: ", "ARX initialization and model function f are size incomptible.")
            exit(-1)
        return f@self.params

    def step(self,u):
        self.U = np.roll(self.U,1) # shift to right by 1 position
        self.U[0] = u
        y = self.output()
        self.Y = np.roll(self.Y,1) # shift to right by 1 position
        self.Y[0] = y
        return y 
##########################################################

def generate_u_data1(n, a, b):
    if (a > b):
        print(inspect.currentframe().f_code.co_name, "():", "ERROR: Input parameters sanity check failed.")
        exit(-1)
    return a + (b-a)*np.random.rand(n)
##########################################################

def generate_u_data2(n, m, sigma):
    if (sigma < 0):
        print(inspect.currentframe().f_code.co_name, "(): ", "ERROR Input parameters sanity check failed.")
        exit(-1)
    return np.random.normal(m, sigma, n)
##########################################################

def generate_arx_data(arx,x,sigma):
    """ for given ARX object, generate a n-points data set, each point 
        consisting of a random input and the output. 
        consider sigma noise level on the output """
    output = []
    for i in range(x.shape[0]):
        output.append(arx.step(x[i])+sigma*np.random.randn()) # generate a noisy output 
    return np.array(output)
##########################################################