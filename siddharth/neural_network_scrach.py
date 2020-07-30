# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 02:54:01 2020

@author: leno
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X = np.vstack([(np.random.rand(10,2)*5),(np.random.rand(10,2)*10)])
y = np.hstack([[0]*10,[1]*10])
dataset = pd.DataFrame(X, columns={'X1','X2'})
dataset['Y'] = y

plt.plot(dataset, label='incline label')
plt.legend(['X1','X2','Y'])

z = np.zeros((20,2))
for i in range(20):
    z[i , y[i]] = 1
    
Wi_1 = np.random.randn(3,2)
Bi_1 = np.random.randn(3)
Wi_2 = np.random.randn(3,2)
Bi_2 = np.random.randn(2)

X.dot(Wi_1.T)

def forward_prop(X, Wi_1, Bi_1, Wi_2, Bi_2):
    #first layer
    M = 1/ (1 + np.exp(-(X.dot(Wi_1.T)+ Bi_1)))
    #secomd layer
    A = M.dot(Wi_2) + Bi_2
    expA = np.exp(A)
    y = expA / expA.sum(axis = 1,  keepdims = True)
    return y, M
    
#returns gradients for weight_2
def diff_Wi_2(H, z, y):
    return H.T.dot(z - y)

#returns gradients for weight_2
def diff_Wi_1(X, H, z, output, Wi_2):
    dz = (z - output).dot(Wi_2.T) * H * (1-H)
    return X.T.dot(dz)

#returns derivative of both thebias
def diff_B2(z, y):
    return(z - y).sum(axis = 0)
    
def diff_B1(z, y, Wi_2, H):
    return ((z - y).dot(Wi_2.T) * H * (1 - H)).sum(axis = 0)

learning_rate = 1e-3
for epooch in range(5000):
    output, hidden = forward_prop(X, Wi_1, Bi_1, Wi_2, Bi_2)
    Wi_2 += learning_rate * diff_Wi_2(hidden, z, output)
    Bi_2 += learning_rate * diff_B2(z, output)
    Wi_1 += learning_rate * diff_Wi_1(X, hidden, z, output, Wi_2).T
    Bi_1 += learning_rate * diff_B1(z, output,  Wi_2, hidden)

X_test = np.array([9,9])

hidden_output = 1 / (1 + np.exp(-X_test.dot(Wi_1.T) - Bi_1))
outer_layer_output = hidden_output.dot(Wi_2) + Bi_2
expA = np.exp(outer_layer_output)
y = expA / expA.sum()
print('prob of class 0>>>>>> {} \n prob of class 1>>>>> {}'.format(y[0],y[1]))







