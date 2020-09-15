#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

# Loading data
#data = np.load('data2D.npy')
data = np.load('data100D.npy')
[num_pts, dim] = np.shape(data)

is_valid = True

# For Validation set
if is_valid:
    valid_batch = int(num_pts / 3.0)
    np.random.seed(45689)
    rnd_idx = np.arange(num_pts)
    np.random.shuffle(rnd_idx)
    val_data = data[rnd_idx[:valid_batch]]
    data = data[rnd_idx[valid_batch:]]

# Distance function for K-means
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the squared pairwise distance matrix (NxK)
    
    X_square = tf.reshape( tf.reduce_sum(X*X,axis=1), [-1,1])
    MU_square = tf.transpose( tf.reshape( tf.reduce_sum(MU*MU,axis=1), [-1,1])  )
    dist = X_square + MU_square - 2*tf.matmul(X, tf.transpose(MU))
    return dist
    
    
def buildGraph():
    k = 10
    x = tf.placeholder(tf.float32,[None,dim])
    mu = tf.get_variable("MU",initializer=tf.truncated_normal(shape = (k,dim)))
    
    dist = distanceFunc(x, mu)
    cluster = tf.argmin(input = dist, axis=1)
    loss = tf.reduce_sum(tf.reduce_min(dist, axis=1))
    
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9, beta2=0.99, epsilon=1e-5)
    training_op = optimizer.minimize(loss=loss)
    
    return training_op,x,mu,cluster,loss


# In[2]:



training_op, x,mu,cluster,loss = buildGraph()

init_op = tf.global_variables_initializer()
with tf.Session() as sess: 
    sess.run(init_op)
    
    Train_loss = []
    
    for epoch in range(300):        
        sess.run(training_op, feed_dict={x: data})
            
        e,c = sess.run([loss,cluster], feed_dict={x: data})
        
        Train_loss = np.append(Train_loss,e)
        
        if epoch%50==0:
            print("Iteration: %d, Error: %.4f " %(epoch, e))

        if epoch>=299:
            e2,c2 = sess.run([loss,cluster], feed_dict={x: val_data})
    
    


# In[3]:


import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(Train_loss)

plt.title(label="Loss Curve",loc="center")
#plt.legend(['Train','Valid','Test'])
plt.xlabel('# iteration')
#plt.ylim((0,2))
plt.ylabel('Loss')
plt.grid('on')
plt.show()


# In[4]:


import matplotlib.pyplot as plt

c0 = np.where(c==0)
c1 = np.where(c==1)
c2 = np.where(c==2)
c3 = np.where(c==3)
c4 = np.where(c==4)

print("C0: %.4f, C1: %.4f, C2: %.4f, C3: %.4f, C4: %.4f,"
      %(100*len(c0[0])/len(c), 100*len(c1[0])/len(c), 100*len(c2[0])/len(c), 100*len(c3[0])/len(c), 100*len(c4[0])/len(c)))


fig = plt.figure()
plt.scatter(data[:,0],data[:,1],c=c,cmap='RdYlBu')

plt.title(label="K-Means Cluster Plots@ K=5",loc="center")
plt.xlabel('x location')
#plt.ylim((0,2))
plt.ylabel('y location')
plt.grid('on')
plt.show()


# In[5]:


e2 #5


# In[6]:


e2 #4


# In[7]:


e2 #3


# In[8]:


e2 #2


# In[9]:


e2 #1


# In[ ]:




