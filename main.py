import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


import Network


Model = Network.ConvolutionalNerualNetwork()

df = pd.read_csv('emnist-letters-train.csv') # accessing the csv
td = df.to_numpy() #converting to numpy array

rand = np.random.default_rng() 
Model = Network.ConvolutionalNerualNetwork()


n = 0.1  #learning rate or step size for gradient descent
Gradients = None
count = 1
batch_size = 128 #batch size for mini batch for gradient descent
epochs = 20# epochs
epoch_size = 88799*0.98 # limit of the samples in epoch

Loss_data = []
Accuracy_data = []




#training the model using gradient descent
for i in range(epochs): 
    # setting accuracy counter
    acc = 0
    Losses = 0
    #suffle each epoch
    rand.shuffle(td)

    while count <= epoch_size:
        batch_acc = 0
        batch_loss = 0
        for k in range(batch_size):
            Input = np.rot90(np.flip(td[count][1:].reshape(28,28),1)) 
            Label = np.zeros((26,1))
            lbl = td[count][0]-1
            Label[lbl][0] = 1
            O = Model.forwardPropogation(Input)
            Loss = Network.SSR.SSR(O,Label)
            Losses+=Loss
            batch_loss +=Loss
            #accuracy update
            if np.argmax(O) == lbl:
                    acc+=1
                    batch_acc +=1

            if Gradients !=None:
                    #summing gradients of each sample in mini batch that affects the cost
                Gradients = Network.SumGradients( Network.BackProp(Model,Input,Label),Gradients)
            else:
                Gradients = Network.BackProp(Model,Input,Label)
                
            count+=1
            #print(Gradients)
            
        Network.GradientUpdate(Gradients,Model , n/batch_size) # gradient descent by updating the model
        Gradients = None # resets gradients for next calculation in descent
        print(batch_loss/batch_size , f"--- count {count} ----epoch {i} --- batch_accuracy => {batch_acc/batch_size}" , np.argmax(O) , lbl)
        
    Loss_data.append(Losses/epoch_size) 
    os.system('cls' if os.name == 'nt' else 'clear')
    print(Losses/epoch_size , f"--- epoch {i} --- accuracy => {acc/epoch_size}" , np.argmax(O) , lbl)
    count = 1
    Accuracy_data.append(acc/epoch_size)
print(Loss_data)
print(Accuracy_data)