#!/usr/bin/env python
# coding: utf-8

# In[6]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target

def relu(x):
    #x Nxd
    #out Nxd
    return np.maximum(x,0)

def softmax(z):
    #z NxK
    #out NxK
    N, K = z.shape
    #subtract max to prevent overflow
    z = z - np.amax(z,axis=1).reshape((N,1))
    expo = np.exp(z)
    sumz = np.reshape(np.sum(expo,axis=1),(N,1))
    return expo/sumz 

# X Nxd
# W dxd'
# b 1xd'
def computeLayer(X, W, b):
    return np.matmul(X,W)+b

def averageCE(target, prediction):
    #traget NxK
    #prediction NxK
    N,_ = target.shape
    p = softmax(prediction)
    return -np.sum(np.log(p) * target)/N

def gradCE(target, prediction):
    #traget NxK
    #prediction NxK
    #out NxK
    N,_ = target.shape
    p = softmax(prediction)
    return p - target

def forward_prop(x0, Wh, Wo, bh, bo):
    #x0 Nx784
    #Wh 784x1000
    #Wo 1000xK
    #z1 Nx1000
    #x1 Nx1000
    #z2 NxK
    z1 = computeLayer(x0, Wh, bh)
    x1 = relu(z1)
    
    z2 = computeLayer(x1, Wo, bo)
    return z1, x1, z2

def back_prop(x0, z1, x1, z2, Wh, Wo, y):
    #x0 Nx784
    #z1 Nx1000
    #x1 Nx1000
    #z2 NxK
    #Wh 784x1000
    #Wo 1000xK
    #y  NxK
    #gradbo 1xK
    #gradWo KxK
    #gradbh 1x1000
    #gradWh 784x1000
    N,_ = x0.shape
    delta2 = gradCE(y, z2) #NxK
    
    gradbo = np.mean(delta2,axis=0)
    gradWo = np.matmul(x1.transpose(),delta2)/N
    
    delta1 = np.matmul(delta2,Wo.transpose()) 
    delta1[z1==0] = 0 #Nx1000
    
    gradbh = np.mean(delta1,axis=0)
    gradWh = np.matmul(x0.transpose(),delta1)/N 
    return gradbo,gradWo,gradbh,gradWh

def accuracy(target, z2):
    N,_ = target.shape
    p = softmax(z2)
    y = np.argmax(target, axis = 1)
    y_hat = np.argmax(p, axis = 1)
    return np.sum(y==y_hat)/N


# In[7]:


trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
newtrain, newvalid, newtest = convertOneHot(trainTarget, validTarget, testTarget)

trainData = trainData.reshape((10000,784))
validData = validData.reshape((6000,784))
testData = testData.reshape((2724,784))


# In[11]:


#initialization
x0 = trainData
y = newtrain
Wh = np.random.normal(0, np.sqrt(2/1784), (784,1000))
Wo = np.random.normal(0, np.sqrt(2/1010), (1000,10))
bh = np.random.normal(0, np.sqrt(2/1784), (1,1000))
bo = np.random.normal(0, np.sqrt(2/1010), (1,10))
v_Wh = np.ones((784,1000))*0.00001
v_Wo = np.ones((1000,10))*0.00001
v_bh = np.ones((1,1000))*0.00001
v_bo = np.ones((1,10))*0.00001
gamma = 0.99
alpha = 0.01
train_result, valid_result, test_result = [],[],[]
train_acc, valid_acc, test_acc = [],[],[]

#training
for i in range(200):
    #calculations
    z1, x1, z2 = forward_prop(x0, Wh, Wo, bh, bo)
    _, _, valid_z2 = forward_prop(validData, Wh, Wo, bh, bo)
    _, _, test_z2 = forward_prop(testData, Wh, Wo, bh, bo)
    gradbo,gradWo,gradbh,gradWh = back_prop(x0, z1, x1, z2, Wh, Wo, y)
    
    #updates
    v_Wh = gamma * v_Wh + alpha * gradWh
    v_Wo = gamma * v_Wo + alpha * gradWo
    v_bh = gamma * v_bh + alpha * gradbh
    v_bo = gamma * v_bo + alpha * gradbo
    Wh = Wh - v_Wh
    Wo = Wo - v_Wo
    bh = bh - v_bh
    bo = bo - v_bo
    
    #results
    if i%10==0:
        train_result = np.append(train_result,averageCE(y, z2))
        valid_result = np.append(valid_result,averageCE(newvalid, valid_z2))
        test_result = np.append(test_result,averageCE(newtest, test_z2))
        train_acc = np.append(train_acc,accuracy(y, z2))
        valid_acc = np.append(valid_acc,accuracy(newvalid, valid_z2))
        test_acc = np.append(test_acc,accuracy(newtest, test_z2))
    
    
        print(i)
        print(train_result[-1],valid_result[-1],test_result[-1])
        print(train_acc[-1],valid_acc[-1],test_acc[-1])


# In[12]:


import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(np.linspace(0,200,len(train_result)),train_result,alpha = 1)
plt.plot(np.linspace(0,200,len(valid_result)),valid_result,alpha = 1)
plt.plot(np.linspace(0,200,len(test_result)),test_result,alpha = 1)
plt.title(label="Loss Curve",loc="center")
plt.legend(['Train','Valid','Test'])
plt.xlabel('# iteration')
#plt.ylim((2.301,2.304))
plt.ylabel('Loss')
plt.grid('on')
plt.show()


# In[13]:


import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(np.linspace(0,200,len(train_acc)),train_acc,alpha = 1)
plt.plot(np.linspace(0,200,len(valid_acc)),valid_acc,alpha = 1)
plt.plot(np.linspace(0,200,len(test_acc)),test_acc,alpha = 1)
plt.title(label="Accuracy Curve",loc="center")
plt.legend(['Train','Valid','Test'])
plt.xlabel('# iteration')
#plt.ylim((2.301,2.304))
plt.ylabel('Accuracy')
plt.grid('on')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




