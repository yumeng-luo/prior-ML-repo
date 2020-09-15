#!/usr/bin/env python
# coding: utf-8

#no
# yes


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


# In[2]:


trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
newtrain, newvalid, newtest = convertOneHot(trainTarget, validTarget, testTarget)

trainData = trainData.reshape((10000,784))
validData = validData.reshape((6000,784))
testData = testData.reshape((2724,784))


# In[81]:


#initialization
x0 = trainData
y = newtrain
unit = 1000
Wh = np.random.normal(0, np.sqrt(2/(784+unit)), (784,unit))
Wo = np.random.normal(0, np.sqrt(2/(10+unit)), (unit,10))
bh = np.random.normal(0, np.sqrt(2/(784+unit)), (1,unit))
bo = np.random.normal(0, np.sqrt(2/(10+unit)), (1,10))
v_Wh = np.ones((784,unit))*0.00001
v_Wo = np.ones((unit,10))*0.00001
v_bh = np.ones((1,unit))*0.00001
v_bo = np.ones((1,10))*0.00001
gamma = 0.99
alpha = 0.05
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
print(i)
print(train_result[-1],valid_result[-1],test_result[-1])
print(train_acc[-1],valid_acc[-1],test_acc[-1])


# In[155]:


import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(np.linspace(0,200,len(train_result)),train_result,'-*',alpha = 1)
plt.plot(np.linspace(0,200,len(valid_result)),valid_result,'--',alpha = 1)
plt.plot(np.linspace(0,200,len(test_result)),test_result,'-.',alpha =1)

plt.title(label="Loss Curve",loc="center")
plt.legend(['Train','Valid','Test'])
plt.xlabel('# iteration')
#plt.ylim((0.25,0.75))
plt.ylabel('Loss')
plt.grid('on')
plt.show()

fig = plt.figure()
plt.plot(np.linspace(0,200,len(train_result)),train_result,'-*',alpha = 1)
plt.plot(np.linspace(0,200,len(valid_result)),valid_result,'--',alpha = 1)
plt.plot(np.linspace(0,200,len(test_result)),test_result,'-.',alpha =1)
plt.scatter(80,train_result[8],marker='X',color='r',alpha = 1)
plt.scatter(80,valid_result[8],marker='X',color='r',alpha = 1)
plt.scatter(80,test_result[8],marker='X',color='r',alpha = 1)
plt.arrow(80, train_result[8]-0.2, dx=0, dy=0.1,head_width=5, head_length=0.1, fc='k', ec='k')
plt.annotate('early stopping point', xy=(50,train_result[8]-0.25))
plt.title(label="Loss Curve - Stop at iteration 80",loc="center")
plt.legend(['Train','Valid','Test'])
plt.xlabel('# iteration')
plt.ylim((0,1))
plt.ylabel('Loss')
plt.grid('on')
plt.show()


# In[156]:


import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(np.linspace(0,200,len(train_acc)),train_acc,'-*',alpha = 1)
plt.plot(np.linspace(0,200,len(valid_acc)),valid_acc,'--',alpha = 1)
plt.plot(np.linspace(0,200,len(test_acc)),test_acc,'-.',alpha = 1)
plt.title(label="Accuracy Curve",loc="center")
plt.legend(['Train','Valid','Test'])
plt.xlabel('# iteration')
#plt.ylim((0.85,0.93))
plt.ylabel('Accuracy')
plt.grid('on')
plt.show()

fig = plt.figure()
plt.plot(np.linspace(0,200,len(train_acc)),train_acc,'-*',alpha = 1)
plt.plot(np.linspace(0,200,len(valid_acc)),valid_acc,'--',alpha = 1)
plt.plot(np.linspace(0,200,len(test_acc)),test_acc,'-.',alpha = 1)
plt.scatter(80,train_acc[8],marker='X',color='r',alpha = 1)
plt.scatter(80,valid_acc[8],marker='X',color='r',alpha = 1)
plt.scatter(80,test_acc[8],marker='X',color='r',alpha = 1)
plt.arrow(80, train_acc[8]+0.04, dx=0, dy=-0.03,head_width=5, head_length=0.01, fc='k', ec='k')
plt.annotate('early stopping point', xy=(50,train_acc[8]+0.04))
plt.title(label="Accuracy Curve - Stop at iteration 80",loc="center")
plt.legend(['Train','Valid','Test'])
plt.xlabel('# iteration')
plt.ylim((0.85,1))
plt.ylabel('Accuracy')
plt.grid('on')
plt.show()


# In[143]:


import matplotlib.pyplot as plt
fig = plt.figure()
accuracy = np.array([0.888766519824,0.886196769457,0.887665198238,0.881791483113])
loss = np.array([0.752694646046,0.760072781112,0.825005188795,0.861162421059])
plt.plot(np.array([100,500,1000,2000]),accuracy,alpha = 1)
plt.scatter(np.array([100,500,1000,2000]),accuracy,marker='X',alpha = 1)
plt.title(label="Final Test Accuracy",loc="center")
#plt.legend(['Train','Valid','Test'])
plt.xlabel('# hidden unit')
plt.ylim((0.85,0.93))
plt.ylabel('Accuracy')
plt.grid('on')
plt.show()

fig = plt.figure()
plt.plot(np.array([100,500,1000,2000]),loss,alpha = 1)
plt.scatter(np.array([100,500,1000,2000]),loss,marker='X',alpha = 1)
plt.title(label="Final Test Loss",loc="center")
#plt.legend(['Train','Valid','Test'])
plt.xlabel('# hidden unit')
#plt.ylim((0.85,0.93))
plt.ylabel('Loss')
plt.grid('on')
plt.show()


# In[1]:


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

def buildGraph():
    reg = tf.Variable(0.0)
    p = tf.Variable(0.5)
    #input layer
    x0 = tf.placeholder(tf.float32,[None,28,28,1])
    y = tf.placeholder(tf.float32,[None,10])
    
    # 3x3 convoluntion layer
    
    W0 = tf.get_variable("W0",shape=[3,3,1,32],dtype=tf.float32,
                         initializer=tf.contrib.layers.xavier_initializer())
    b0 = tf.get_variable("b0",shape=32,dtype=tf.float32,
                         initializer=tf.contrib.layers.xavier_initializer())
    
    x1 = tf.nn.conv2d(x0,W0,strides=[1,1,1,1],padding='SAME')
    x1 = tf.nn.bias_add(x1,b0)
    
    #ReLU layer
    x2 = tf.nn.relu(x1)
    
    #batch normalization
    mean, variance = tf.nn.moments(x2,axes=[0])
    x3 = tf.nn.batch_normalization(x2,mean,variance,None,None,1e-5)
    
    #max pooling
    x4 = tf.nn.max_pool(x3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    
    #flatten layer
    x5 = tf.reshape(x4,[-1,6272])
    
    #fully connect layer
    W5 = tf.get_variable('W5',shape=[6272,784],dtype=tf.float32,
                        regularizer=tf.contrib.layers.l2_regularizer(reg))
    b5 = tf.get_variable('b5',shape=784,dtype=tf.float32,
                        regularizer=tf.contrib.layers.l2_regularizer(reg))
    x6a = tf.matmul(x5,W5) + b5
    
    #drop out
    x6 = tf.nn.dropout(x6a,p)
    
    #ReLU layer
    x7 = tf.nn.relu(x6)
    
    #fully connect layer
    W7 = tf.get_variable('W7',shape=[784,10],dtype=tf.float32,
                        regularizer=tf.contrib.layers.l2_regularizer(reg))
    b7 = tf.get_variable('b7',shape=10,dtype=tf.float32,
                        regularizer=tf.contrib.layers.l2_regularizer(reg))
    x8 = tf.matmul(x7,W7) + b7
    
    #softmax layer
    x9 = tf.nn.softmax(x8)
    
    #CE loss
    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=x8))
    
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    training_op = optimizer.minimize(loss=error)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(x9,1),tf.argmax(y,1)),tf.float32))
    return error, training_op, accuracy, x0, y


# In[2]:


trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
newtrain, newvalid, newtest = convertOneHot(trainTarget, validTarget, testTarget)

trainData = trainData.reshape((10000,28,28,1))
validData = validData.reshape((6000,28,28,1))
testData = testData.reshape((2724,28,28,1))

batch_size = 32
num_batch = int(np.ceil(10000/batch_size))

error, training_op, accuracy, x0, y = buildGraph()
init_op = tf.global_variables_initializer()
with tf.Session() as sess: 
    sess.run(init_op)
    
    Train_loss, ValidLoss, TestLoss = [],[],[]
    Train_acc, Valid_acc, Testacc = [],[],[]
    
    for epoch in range(50):        
        #shuffle data
        data, label = shuffle(trainData, newtrain)
        
        for i in range(num_batch):
            #devide data sets into #batch size
            if i is num_batch-1:
                x = data[i*batch_size:-1,:]
                y_target = label[i*batch_size:-1,:]
            else:
                x = data[i*batch_size:(i+1)*batch_size,:]
                y_target = label[i*batch_size:(i+1)*batch_size,:]
            
            sess.run(training_op, feed_dict={x0: x, y: y_target})
            
        el,ea = sess.run([error,accuracy], feed_dict={x0: trainData, y: newtrain})
        vl,va = sess.run([error,accuracy], feed_dict={x0: validData, y: newvalid})
        tl,ta = sess.run([error,accuracy], feed_dict={x0: testData, y: newtest})
        
        Train_loss = np.append(Train_loss,el)
        Train_acc = np.append(Train_acc,ea)    
        ValidLoss = np.append(ValidLoss,vl)
        Valid_acc = np.append(Valid_acc,va)
        TestLoss = np.append(TestLoss,tl)
        Testacc = np.append(Testacc,ta)
        print("Iteration: %d, Error: %.4f %.4f %.4f Accuracy: %.4f %.4f %.4f" %(epoch, el,vl,tl,ea,va,ta))


# In[5]:


import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(np.linspace(0,50,len(Train_loss)),Train_loss,alpha = 1)
plt.plot(np.linspace(0,50,len(ValidLoss)),ValidLoss,alpha = 1)
plt.plot(np.linspace(0,50,len(TestLoss)),TestLoss,alpha =1)

plt.title(label="Loss Curve @ p=0.5",loc="center")
plt.legend(['Train','Valid','Test'])
plt.xlabel('# iteration')
#plt.ylim((0.25,0.75))
plt.ylabel('Loss')
plt.grid('on')
plt.show()



fig = plt.figure()
plt.plot(np.linspace(0,50,len(Train_acc)),Train_acc,alpha = 1)
plt.plot(np.linspace(0,50,len(Valid_acc)),Valid_acc,alpha = 1)
plt.plot(np.linspace(0,50,len(Testacc)),Testacc,alpha =1)

plt.title(label="Accuracy Curve @ p=0.5",loc="center")
plt.legend(['Train','Valid','Test'])
plt.xlabel('# iteration')
plt.ylim((0.9,1))
plt.ylabel('Accuracy')
plt.grid('on')
plt.show()


# In[ ]:





# In[ ]:





# In[158]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




