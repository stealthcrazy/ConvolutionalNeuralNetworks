import numpy as np
import scipy as sc


class ConvolutionalLayer:
    def __init__(self,kernelDimension):
        self.kernelDimension = kernelDimension # size of kernel filter for convolution
        self.stride     = 0          # for Future Use
        self.kernel = np.random.normal(loc = 0 , scale = 1e-1 , size = (self.kernelDimension,self.kernelDimension))
        self.Bias = 0
    def forward(self,X):
        self.A = X
        self.Out = sc.signal.correlate2d(X, self.kernel, mode='valid')+ (np.ones((X.shape[0] -  self.kernelDimension +1, X.shape[1] -  self.kernelDimension +1))*self.Bias)
        return self.Out
    def backward(self,S):
        self.S = S
        self.dC_dB = np.sum(self.S)
        self.dC_dF = sc.signal.correlate2d(self.A, self.S, mode='valid')
        
class LinearLayer:
    def __init__(self,InDim,OutDim):
        self.WeightDimension = (OutDim,InDim)
        self.BiasDimension =   OutDim
        self.Weight = np.random.normal(loc = 0 , scale = 2/InDim , size = self.WeightDimension )
        self.Bias = np.zeros(shape = (self.BiasDimension,1))
    def forward(self,X):
        self.A = X
        self.Out = (self.Weight @ X) + self.Bias
        return self.Out
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
        Y = X.copy()
        if derivative == True:
            Y[Y<0] = 0
            Y[Y>0] = 1
            return Y
        Y[Y<0] = 0
        return Y
    @staticmethod
    def softmax(X, derivative = False):
        if derivative == True:
            return np.diag(X.T[0])- X@X.T
        return np.exp(X) / np.sum(np.exp(X))




class SSR:
    @staticmethod
    def SSR(O , X ,derivative = False):
        if derivative == True:
            return 2*(O - X)
        return np.linalg.norm(O-X)**2


class CrossEntropyLoss:
    @staticmethod
    def CrossEntropyLoss(P,Q,derivative = False):
        if derivative == True:
            return Q-P
        return -1*np.dot(P.reshape(-1),np.log(Q).reshape(-1))



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
        self.Out = self.Pooled       
        return self.Out
    def backward(self,S):
        self.dC_dPool = np.zeros(( self.XHeight,self.XWidth ))
        for i in range(0,self.PooledHeight):
            for j in range(0,self.PooledWidth):
                M = self.A[i:i+self.kernelDimension,j:j+self.kernelDimension]
                m,n= np.unravel_index(np.argmax(M), M.shape)
                self.dC_dPool[i+m,j+n] += S[i,j]
        return self.dC_dPool
                
class ConvolutionalNerualNetwork():
    def __init__(self):
        # defining hyper parameters for each layer
        # intialising layers and pooling
        self.cv1 = ConvolutionalLayer(kernelDimension=5)
        self.pl1 = MaxPool(kernelDimension=3)
        self.cv2 = ConvolutionalLayer(kernelDimension=3)
        self.pl2 = MaxPool(kernelDimension=3)

        self.l1 = LinearLayer(InDim=324,OutDim=128)
        self.l2 = LinearLayer(InDim=128,OutDim=64)
        self.l3 = LinearLayer(InDim=64,OutDim=26)






    def forwardPropogation(self,INPUT):
        #first Convultion + RELU activation
        self.Z1 = self.cv1.forward(INPUT)
        self.A1 = Activation.ReLU(self.Z1)
        # Max Pooling 1
        self.P1 = self.pl1.forward(self.A1)

        # second convolution + RELU 
        self.Z2 =  self.cv2.forward(self.P1)
        self.A2 = Activation.ReLU(self.Z2)
        # Max Pooling 2
        self.P2 =  self.pl2.forward(self.A2)

        # flattening
        self.F1 = self.P2.flatten().reshape((-1,1))

        # Fully connected Neural Network
        self.Z3 =  self.l1.forward(self.F1)
        self.A3 = Activation.ReLU(self.Z3)

        self.Z4 =  self.l2.forward(self.A3)
        self.A4 = Activation.ReLU(self.Z4)

        self.Z5 =  self.l3.forward(self.A4)
        self.A5 = Activation.softmax(self.Z5)


        return self.A5

def BackProp(Model: ConvolutionalNerualNetwork,Input,Label,): # manual Back Propogation static Graph

    Gradients = {}


    #dL_dA5 = SSR.SSR(O=Model.A5,X=Label,derivative=True) # using  Sum of Squared Residuals
    #dA5_dZ5 = Activation.ReLU(X=Model.Z5,derivative=True)
    #S5 = np.multiply(dL_dA5,dA5_dZ5)
    #using CrossEntropyLoss
    S5= CrossEntropyLoss.CrossEntropyLoss(P=Label,Q=Model.A5 ,derivative=True)

    

    Model.l3.backward(S5)
    Gradients["W3"] = Model.l3.dC_dW 
    Gradients["B3"] = Model.l3.dC_dB

    dL_dA4 = (Model.l3.Weight.T @S5)
    dA4_dZ4 = Activation.ReLU(X=Model.Z4,derivative=True)
    S4 = np.multiply(dL_dA4,dA4_dZ4)

    Model.l2.backward(S4)
    Gradients["W2"] = Model.l2.dC_dW 
    Gradients["B2"] = Model.l2.dC_dB

    dL_dA3 = (Model.l2.Weight.T @S4)
    dA3_dZ3 = Activation.ReLU(X=Model.Z3,derivative=True)
    S3 = np.multiply(dL_dA3,dA3_dZ3)

    Model.l1.backward(S3)
    Gradients["W1"] = Model.l1.dC_dW 
    Gradients["B1"] = Model.l1.dC_dB

    dL_dF1 = (Model.l1.Weight.T @S3).reshape((18,18)) 
    dL_dA2 = Model.pl2.backward(dL_dF1)
    dA2_dZ2 = Activation.ReLU(X=Model.Z2,derivative=True)
    S2 =  np.multiply(dL_dA2,dA2_dZ2)

    Model.cv2.backward(S2)
    Gradients["F2"] = Model.cv2.dC_dF
    Gradients["FB2"] = Model.cv2.dC_dB
    dL_dP1 = sc.signal.correlate2d(S2, Model.cv2.kernel, mode='full')
    dL_dA1 = Model.pl1.backward(dL_dP1)
    dA1_dZ1 = Activation.ReLU(X=Model.Z1,derivative=True)
    S1 = np.multiply(dL_dA1,dA1_dZ1)
    Model.cv1.backward(S1)
    Gradients["F1"] = Model.cv1.dC_dF
    Gradients["FB1"] = Model.cv1.dC_dB
    return Gradients

def GradientUpdate(Gradients,Model:ConvolutionalNerualNetwork,n):

    Model.l3.Weight -= n*Gradients["W3"]
    Model.l3.Bias -= n*Gradients["B3"]
    Model.l2.Weight -= n*Gradients["W2"]
    Model.l2.Bias -= n*Gradients["B2"]
    Model.l1.Weight -= n*Gradients["W1"]
    Model.l1.Bias -= n*Gradients["B1"]
    Model.cv2.kernel -= n*Gradients["F2"]
    Model.cv2.Bias -= n*Gradients["FB2"]
    Model.cv1.kernel -= n*Gradients["F1"]
    Model.cv1.Bias -= n*Gradients["FB1"]

def SumGradients(Gradients , OldGrad):
    # manually summing the gradients for the minibatches
     OldGrad["W3"] +=  Gradients["W3"]
     OldGrad["B3"] += Gradients["B3"]
     OldGrad["W2"] +=  Gradients["W2"]
     OldGrad["B2"] += Gradients["B2"]
     OldGrad["W1"] +=  Gradients["W1"]
     OldGrad["B1"] +=  Gradients["B1"]
     OldGrad["F2"] +=  Gradients["F2"]
     OldGrad["FB2"] +=  Gradients["FB2"]
     OldGrad["F1"] +=  Gradients["F1"]
     OldGrad["FB1"] +=  Gradients["FB1"]

     return OldGrad



    






    
    








