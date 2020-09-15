#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def loadData():
#notMNIST.npz also found in same directory :)
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
x1 = np.reshape(trainData, (3500,-1))
y1 = (trainTarget).astype(float)

x2 = np.reshape(validData, (100,-1))
y2 = (validTarget).astype(float)

x3 = np.reshape(testData, (145,-1))
y3 = (testTarget).astype(float)


# In[2]:


def MSE(W, b, x, y, reg):
    N,d = x.shape
    assert d == len(W),"W & X size mismatch"
    W_error = reg/2 * np.square(np.linalg.norm(W))
    MSE_error = 0
    for i in range(N):
        y_hat = np.dot(W,x[i]) + b
        MSE_error += np.square(y_hat - y[i])
    error = MSE_error/N + W_error
    return error


###########################
# x (3500, 28, 28) input data matrix -> (3500, 784) N X d
# y (3500) true labels
# W (784) weight vector
# b float bias
# reg float regularization parameter
#
# gradient_W ,gradient_b float
# gradient_W = sigma( 2/N * xT * (- y + WTx + b) ) + reg * W
# gradient_b = sigma( 2/N * (-y + WTx + b) )
###########################
def gradMSE(W, b, x, y, reg):
    N,d = x.shape
    assert d == len(W),"W & X size mismatch"

    err = np.dot(x,W)+b-y
    gradient_b = 2*np.sum(err)/N
    gradient_W = 2*np.dot(np.transpose(x),err)/N+ reg*W

    
    return gradient_W, gradient_b

def estimate(W, b, x):
    return (np.matmul(x,W)+b)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def crossEntropyLoss(W, b, x, y, reg):
    N,_ = x.shape
    W_error = reg/2 * np.linalg.norm(W,ord=2)
    y_hat = sigmoid1( estimate1(W, b, x) )
    D_error = np.matmul(np.multiply(-1,y),np.log(y_hat)) - np.matmul((1-y),(np.log(1-y_hat)))
    return np.sum(D_error)/N + W_error


def gradCE(W, b, x, y, reg):
    N,_ = x.shape
    y_hat = sigmoid ( estimate(W, b, x) )
    grad_W = np.matmul(x.transpose(), y_hat - y)/N + reg * W
    grad_b = np.sum(y_hat - y)/N
    return grad_W, grad_b

def accuracy(W, b, x, y):
    N = y.shape
    y_hat = sigmoid( estimate(W, b, x) )
    result = np.zeros(N)
    result[y_hat>=0.5] = 1
    accuracy = 1 - np.sum(abs(result - y_hat))/N
    return accuracy
                     
def accuracy2(W, b, x, y):
    N = y.shape
    y_hat = estimate(W, b, x)
    result = np.zeros(N)
    result[y_hat>=0.5] = 1
    accuracy = 1 - np.sum(abs(result - y_hat))/N
    return accuracy

def grad_descent(W, b, x, y, alpha, epochs, reg, error_tol, 
                 ValidData = None, ValidLabel = None, TestData = None, TestLabel = None, lossType="MSE"):
    Train_error = []
    Valid_error = []
    Test_error = []
    Train_Accuracy = []
    Valid_Accuracy = []
    Test_Accuracy = []
    
    #initial
    current_W = W
    current_b = b
    #gradient descent
    for i in range(epochs):
        
        if lossType == "MSE":
            grad_W, grad_b = gradMSE(current_W, current_b, x, y, reg)
        else:
            grad_W, grad_b = gradCE(current_W, current_b, x, y, reg)
        
        next_W = current_W - alpha * grad_W 
        next_b = current_b - alpha * grad_b
        
        step = np.sum(abs(next_W - current_W)) + abs(next_b - current_b)
       
        if i%10 == 0:
            if lossType == "MSE":
                Train_error = np.append(Train_error,MSE(current_W, current_b, x, y, reg))
                Train_Accuracy = np.append(Train_Accuracy,accuracy2(current_W, current_b, x, y))
                if ValidData is not None and ValidLabel is not None:
                    Valid_error = np.append(Valid_error,MSE(current_W, current_b, 
                                                            ValidData, ValidLabel, reg))
                    Valid_Accuracy = np.append(Valid_Accuracy,accuracy2(current_W, current_b, 
                                                                       ValidData, ValidLabel))
                if TestData is not None and TestLabel is not None:
                    Test_error = np.append(Test_error,MSE(current_W, current_b, 
                                                            TestData, TestLabel, reg))
                    Test_Accuracy = np.append(Test_Accuracy,accuracy2(current_W, current_b, 
                                                                       TestData, TestLabel))
            else:
                Train_error = np.append(Train_error,crossEntropyLoss(current_W, current_b, x, y, reg))
                Train_Accuracy = np.append(Train_Accuracy,accuracy(current_W, current_b, x, y))
                if ValidData is not None and ValidLabel is not None:
                    Valid_error = np.append(Valid_error,crossEntropyLoss(current_W, current_b, 
                                                            ValidData, ValidLabel, reg))
                    Valid_Accuracy = np.append(Valid_Accuracy,accuracy(current_W, current_b, 
                                                                       ValidData, ValidLabel))
                if TestData is not None and TestLabel is not None:
                    Test_error = np.append(Test_error,crossEntropyLoss(current_W, current_b, 
                                                            TestData, TestLabel, reg))
                    Test_Accuracy = np.append(Test_Accuracy,accuracy(current_W, current_b, 
                                                                       TestData, TestLabel))
        if abs(step) <= error_tol:
            #convergence break
            break
        current_W = next_W
        current_b = next_b
    return current_W, current_b, Train_error, Valid_error, Test_error, Train_Accuracy, Valid_Accuracy, Test_Accuracy


# In[3]:


def buildGraph(loss="MSE",opt="Adam", beta1=None, beta2=None, epsi=None):
    #Initialize weight and bias tensors
    tf.set_random_seed(421)
    W = tf.Variable(tf.truncated_normal(shape=[784,1], stddev=0.5), name='weights')
    b = tf.Variable(0.0, name='biases')
    X = tf.placeholder(tf.float32, [None, 784], name='input_x')
    y = tf.placeholder(tf.float32, [None, 1], name='target_y')
    reg = tf.placeholder(tf.float32, name='reg')
    alpha = tf.Variable(0.001, name='alpha')
    
    if loss == "MSE":
        y_hat = tf.add(tf.matmul(X, W),b)
        ld = tf.reduce_mean(tf.square(y_hat - y))
        wd = reg/2 * tf.reduce_sum(tf.square(W))
        error = ld + wd

    elif loss == "CE":
        ld = tf.losses.sigmoid_cross_entropy(y, tf.add(tf.matmul(X, W),b))
        y_hat = tf.sigmoid(tf.add(tf.matmul(X, W),b))
        wd = reg/2 * tf.reduce_sum(tf.square(W))
        error = ld + wd
    
    y_predict = tf.to_float(tf.greater(y_hat,0.5))
    accuracy = 1 - tf.reduce_mean(tf.abs(y_predict-y))
    
    if opt == "Adam":
        if beta1 is not None:
            optimizer = tf.train.AdamOptimizer(learning_rate=alpha,beta1=beta1)
        elif beta2 is not None:
            optimizer = tf.train.AdamOptimizer(learning_rate=alpha,beta2=beta2)
        elif epsi is not None:
            optimizer = tf.train.AdamOptimizer(learning_rate=alpha,epsilon=epsi)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=alpha)
    else:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    training_op = optimizer.minimize(loss=error)

    
    return W, b, X, reg, y_predict, y, error, training_op, accuracy


# In[4]:


batch_size = 500
num_batch = int(np.ceil(3500/batch_size))
#data contains both x and y, N x d+1 size. 
data = np.append(x1, trainTarget.astype(float), axis=1)

W, b, X, reg, y_predict, y, error, training_op, accuracy= buildGraph("MSE","Adam")
init_op = tf.global_variables_initializer()
with tf.Session() as sess: 
    sess.run(init_op)
    
    Train_loss, ValidLoss, TestLoss = [],[],[]
    Train_acc, Valid_acc, Testacc = [],[],[]
    
    for epoch in range(700):        
        #shuffle data
        np.random.shuffle(data)
        
        for i in range(num_batch):
            #devide data sets into #batch size
            if i is num_batch-1:
                x = data[i*batch_size:-1,0:784]
                y_target = data[i*batch_size:-1,784:785]
            else:
                x = data[i*batch_size:(i+1)*batch_size,0:784]
                y_target = data[i*batch_size:(i+1)*batch_size,784:785]
            
            sess.run(training_op, feed_dict={X: x, y: y_target, reg:0.1})
            
        el,ea = sess.run([error,accuracy], feed_dict={X: x1, y: y1, reg:0.1})
        vl,va = sess.run([error,accuracy], feed_dict={X: x2, y: y2, reg:0.1})
        tl,ta = sess.run([error,accuracy], feed_dict={X: x3, y: y3, reg:0.1})
        
        Train_loss = np.append(Train_loss,el)
        Train_acc = np.append(Train_acc,ea)    
        ValidLoss = np.append(ValidLoss,vl)
        Valid_acc = np.append(Valid_acc,va)
        TestLoss = np.append(TestLoss,tl)
        Testacc = np.append(Testacc,ta)
        
        if epoch%50==0:
            print("Iteration: %d, Error: %.4f Accuracy: %.4f" %(epoch, el,ea))


# In[ ]:




