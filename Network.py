import numpy as np
import scipy as sc

class ConvolutionalLayer:
    def __init__(self,kernelDimension):
        self.kernelDimension = kernelDimension # size of kernel filter for convolution
        self.stride     = 0          # for Future Use
        self.kernel = np.random.normal(loc = 0 , scale = 1 , size = (self.kernelDimension,self.kernelDimension))
    
    def forward(self,X):
        self.A = X
        return sc.signal.correlate2d(X, self.kernel, mode='valid')
    def backward(self,S):
        self.S = S
        self.dC_dF = sc.signal.correlate2d(self.A, self.S, mode='valid')
        
class LinearLayer:
    def __init__(self,InDim,OutDim):
        self.WeightDimension = (OutDim,InDim)
        self.BiasDimension =   (OutDim,1)
        self.Weight = np.random.normal(loc = 0 , scale = 1 , size = self.WeightDimension )
        self.Bias = np.zeros(self.BiasDimension)
    def forward(self,X):
        self.A = X
        return (self.Weight @ X) + self.Bias
    def backward(self,S):
        self.S = S
        self.dC_dW = self.S@(self.A.T)
        self.dC_dB = self.S

class Activation:
    @staticmethod
    def sig(M , derivative = False):
        if derivative == True:
            return Activation.sig(M)*(1-Activation.sig(M))
        return 1/(np.exp(-M)+1)
    
    @staticmethod
    def ReLU(X, derivative = False):
        if derivative == True:
            X[X<0] = 0
            X[X>0] = 1
            return X
        X[X<0] = 0
        return X





class SSR:
    @staticmethod
    def SSR(O , X ,derivative = False):
        if derivative == True:
            return 2*(O - X)
        return np.linalg.norm(O-X)**2

class MaxPool:
    def __init__(self,kernelDimension):
        self.kernelDimension = kernelDimension
    def forward(self,X):
        self.A = X
        self.XHeight = X.shape[0]
        self.XWidth = X.shape[1]
        self.PooledHeight = X.shape[0] -  self.kernelDimension +1
        self.PooledWidth =  X.shape[1] -  self.kernelDimension +1
        self.Pooled = np.zeros((self.PooledHeight,self.PooledWidth))
        
        for i in range(0,self.PooledHeight):
            for j in range(0,self.PooledWidth):
                S = X[i:i+self.kernelDimension,j:j+self.kernelDimension]
                self.Pooled[i,j] =  np.max(S)
                
        return self.Pooled
    def backward(self):
        self.dC_dPool = np.zeros(( self.XHeight,self.XWidth ))
        for i in range(0,self.PooledHeight):
            for j in range(0,self.PooledWidth):
                S = self.A[i:i+self.kernelDimension,j:j+self.kernelDimension]
                m,n= np.unravel_index(np.argmax(S), S.shape)
                self.dC_dPool[i+m][j+n] += 1
        return self.dC_dPool
                
        
