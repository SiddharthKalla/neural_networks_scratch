#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from im2col import *
class ReLU():
    def __init__(self):
        self.params = []

    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, dout):
        dX = dout.copy()
        dX[self.X <= 0] = 0
        return dX, []


class sigmoid():
    def __init__(self):
        self.params = []

    def forward(self, X):
        out = 1.0 / (1.0 + np.exp(X))
        self.out = out
        return out

    def backward(self, dout):
        dX = dout * self.out * (1 - self.out)
        return dX, []


class tanh():
    def __init__(self):
        self.params = []

    def forward(self, X):
        out = np.tanh(X)
        self.out = out
        return out

    def backward(self, dout):
        dX = dout * (1 - self.out**2)
        return dX, []


# In[ ]:




