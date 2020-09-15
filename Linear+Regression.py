
# coding: utf-8

# In[2]:

import numpy as np
with np.load('notMNIST.npz') as data :
    Data, Target = data ['images'], data['labels']
    posClass = 2
    negClass = 9
    dataIndx = (Target==posClass) + (Target==negClass)
    Data = Data[dataIndx]/255.
    Target = Target[dataIndx].reshape(-1, 1)
    Target[Target==posClass] = 1
    Target[Target==negClass] = 0
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data, Target = Data[randIndx], Target[randIndx]
    trainData, trainTarget = Data[:3500], Target[:3500]
    validData, validTarget = Data[3500:3600], Target[3500:3600]
    testData, testTarget = Data[3600:], Target[3600:]


# In[7]:

##########################
# x (3500, 28, 28) input data matrix -> (3500, 784) N X d
# y (3500) true labels
# W (784) weight vector
# b float bias
# reg float regularization parameter
#
# error = MSE_error + W_error
#       = sigma(1/N * || y_hat - yi|| ^2) + reg/2 * ||W||^2
#########################
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
def grad_MSE(W, b, x, y, reg):
    N,d = x.shape
    assert d == len(W),"W & X size mismatch"
    
    gradient_W = reg * W
    gradient_b = 0
    for i in range(N):
        gradient_W += 2/N * x[i] * (-y[i] + np.dot(W,x[i]) + b )
        gradient_b += 2/N * (-y[i] +  np.dot(W,x[i]) + b)
    
    return gradient_W, gradient_b


# In[17]:

###########################
# x (3500, 784) input data matrix 
# y (3500) true labels
# W (784) initial weight vector
# b float initial bias
# reg float regularization parameter
# alpha float learning rate
# epochs int32 complete pass
# error_tol 10^-7
#
# W_opt (784) 
# b_opt float
###########################
def grad_descent(W, b, x, y, alpha, epochs, reg, error_tol):
    train_error = []
    test_error = []
    valid_error = []
    
    #initial
    current_W = W
    current_b = b
    #gradient descent
    for i in range(epochs):
        grad_W, grad_b = grad_MSE(current_W, current_b, x, y, reg)
        
        next_W = current_W - alpha * grad_W
        next_b = current_b - alpha * grad_b
        
        step = np.sum(abs(next_W - current_W)) + abs(next_b - current_b)
        #print(MSE(current_W, current_b, x, y, reg))
        if i%2 == 0:
            #print(step)
            train_error = np.append(train_error,MSE(current_W, current_b, x, y, reg))
        if abs(step) <= error_tol:
            print("here!")
            print(i)
            break
        current_W = next_W
        current_b = next_b
    return current_W, current_b, train_error


# In[40]:

#test case of 3 3-d points
#(2,1,8), (5,8,21), (1,4,9)
#green triangle is the correct plot
#red circle is the approximation

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
x = np.array([[2.,1],[5,8],[1,4],[6,9]])
y = np.array([8.,21,9,28])
W = np.array([1.,2])
b = 1
reg = 0.001
error_tol = 0.0000001
epochs = 20000
alpha = 0.01

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:,0], x[:,1], y, marker = '^', c = 'g')
######################################################
W_opt, b_opt, error = grad_descent(W, b, x, y, alpha, epochs, reg, error_tol)
W_opt2, b_opt2,error2 = grad_descent(W, b, x, y, 0.005, epochs, reg, error_tol)
W_opt3, b_opt3,error3 = grad_descent(W, b, x, y, 0.001, epochs, reg, error_tol)

######################################################
xs = np.linspace(0, 10, num=100)
xv, yv = np.meshgrid(xs, xs)
zv = W_opt[0] * xv + W_opt[1] * yv + b_opt
ax.scatter(xv, yv, zv,alpha=0.005)
y_hat = x[:,0] * W_opt[0] + x[:,1] * W_opt[1] + b_opt
#print(y_hat)
ax.scatter(x[:,0], x[:,1], y_hat, marker = 'o', c = 'r')

fig = plt.figure()
plt.scatter(np.linspace(0,epochs,len(error)),error,c='r',marker='^',alpha = 1)
plt.scatter(np.linspace(0,epochs,len(error2)),error2,c='g',marker='*',alpha = 0.1)
plt.scatter(np.linspace(0,epochs,len(error3)),error3,c='b',alpha = 0.1)
plt.show()


# In[45]:

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
x = np.reshape(trainData, (3500,-1))
y = trainTarget.flatten()
#可以改
W = np.ones(784)
b = 0
alpha = 0.000001 # 0.005, 0.001, 0.0001
#题目规定了
reg = 0
error_tol = 0.0000001
epochs = 200


W_opt, b_opt,error = grad_descent(W, b, x, y, alpha, epochs, reg, error_tol)
W_opt2, b_opt2,error2 = grad_descent(W, b, x, y, 0.0005, epochs, reg, error_tol)
W_opt3, b_opt3,error3 = grad_descent(W, b, x, y, 0.0001, epochs, reg, error_tol)

fig = plt.figure()
plt.scatter(np.linspace(0,epochs,len(error)),error,c='r',marker='^',alpha = 1)
plt.scatter(np.linspace(0,epochs,len(error2)),error2,c='g',marker='*',alpha = 1)
plt.scatter(np.linspace(0,epochs,len(error3)),error3,c='b',alpha = 1)
plt.show()


# In[ ]:



