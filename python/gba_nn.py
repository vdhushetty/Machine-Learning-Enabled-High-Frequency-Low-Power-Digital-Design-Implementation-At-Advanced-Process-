# -*- coding: utf-8 -*-
"""
Created on Sun May  1 03:12:26 2022

@author: rmanoha5
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.feature_selection import VarianceThreshold 
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression 
#from statistics import mean
#from statistics import stdev
# Importing the datasets

datasets = pd.read_csv('GBA_gcn.csv')
X = datasets.iloc[:, 5:24].values
y = datasets.iloc[:, 24].values
#print(datasets.corr())
#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 1)

#standardization
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_X.fit(X)
X_train_sc = sc_X.transform(X_train)
X_test_sc = sc_X.transform(X_test)
#Feature selection
vt = VarianceThreshold(threshold=(0.8*(1-0.8)))
vt.fit(X)
X = vt.transform(X)
X_train_sc = vt.transform(X_train_sc)
X_test_sc = vt.transform(X_test_sc)
#Feature selection based on Linear Regression Model
selector = RFE(LinearRegression(), n_features_to_select=10, step=1)
selector.fit(X, y)
X_test_sc = selector.transform(X_test_sc)
X_train_sc = selector.transform(X_train_sc)
#X_test = selector.transform(X_test)
#X_train = selector.transform(X_train)
print(X_train_sc.shape)
print(X_test_sc.shape)
#print(X_train.shape)
#print(X_test.shape)
print('Feature Selection', selector.support_)

###########################################################################
###################################Neural Network##########################
import torch

X_train_nn =  X_train_sc.astype(np.float32)
y_train_nn = y_train.astype(np.float32)
X_test_nn =   X_test_sc.astype(np.float32)
y_test_nn = y_test.astype(np.float32)

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = len(X_train_sc[:,0]), len(X_train_sc[0,:]), 12000, 1
y_train_nn, y_test_nn = y_train_nn.reshape(N,1), y_test.reshape(len(X_test[:,0]),1)
#print(N,D_in,D_out)
X_train_nn = torch.tensor(X_train_nn)

X_test_nn = torch.tensor(X_test_nn)
y_train_nn = torch.tensor(y_train_nn)

y_test_nn = torch.tensor(y_test_nn)
#print(y_train_nn)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU (),
    #torch.nn.Dropout(0.25),
    torch.nn.Linear(H, D_out),
    torch.nn.ReLU()
    )

#   torch.nn.Sigmoid())

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 3e-4
last = 1
error = 1
t = 0

optimizer = torch.optim.Adam(model.parameters(),learning_rate )
start = time.time()
while (t<10000 and abs(error)>0.1):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred_nn = model(X_train_nn)
    
    #print('Y_PRED',y_pred_nn)
    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    loss = loss_fn(y_pred_nn, y_train_nn)
   
    if error < 0 and t >1:
        print('loss',loss.item())
        print('iteration',t)
        print('error',error)
        #learning_rate = learning_rate/10
    '''
    if loss/error > 1000 :
        learning_rate = learning_rate*1.1
        '''
    if t % 500 == 499:
        print('loss',loss.item())
        print('iteration',t)
        print('error',error)        
        
    # Zero the gradients before running the backward pass.
    optimizer.zero_grad()
    #model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()
    error = last - loss.item()
    
    last = loss.item()
    
    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    optimizer.step()
    '''
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
        #
'''
     
    t += 1
##
end = time.time()
print('error:', error)
print('loss', loss.item())
print('End Iteration:', t)
#y_pred = onehtar(y_pred)            
#print(y_pred.detach().numpy())
#print(y.detach().numpy())
#print(y_pred_nn,'\n',y_train_nn)
print("***************************NN************************************")
loss_k  = 0
for i in range(0,len(y_train)):
    loss_k = loss_k +np.sqrt(((y_train_nn[i].detach().numpy() - y_pred_nn[i].detach().numpy())**2))    
print('Train accuracy \n Average PBA vs GBA-PBA',loss_k/len(y_train))
y_pred_nn_t = model(X_test_nn)

loss_k  = 0
for i in range(0,len(y_test)):
    loss_k = loss_k +np.sqrt(((y_test_nn[i].detach().numpy() - y_pred_nn_t[i].detach().numpy())**2))    
print('Test accuracy \n Average PBA vs GBA-PBA',loss_k/len(y_test))
print("Total trainig time of the NN:", float(end - start)/60)
loss = np.sqrt(((y_test_nn.detach().numpy() - y_pred_nn_t.detach().numpy())**2))/2
#################PLOTS############################
plt.ylim([0,1600])
plt.xlim([2000,2100])
plt.plot(y_test_nn.detach().numpy(),color='red',label="Real PBA")
plt.plot(y_pred_nn_t.detach().numpy(),color='blue',label="Predicted PBA")
plt.title("GBA-PBA VS Acutal PBA NN")
plt.xlabel("paths") 
plt.ylabel("delay")
plt.legend()    
plt.figure(figsize=(50,8))
plt.show()

#plt.ylim([-40,70])
plt.xlim([2000,2100])
plt.plot(loss, color= 'yellow', label= "loss")
plt.title("NN Loss")
plt.xlabel("Paths") 
plt.ylabel("Loss")
plt.legend()
plt.figure(figsize=(50,8))
plt.show()

