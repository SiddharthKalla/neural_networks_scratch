#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from im2col import *
class Dropout():

    def __init__(self, prob=0.5):
        self.prob = prob
        self.params = []

    def forward(self, X):
        self.mask = np.random.binomial(1, self.prob, size=X.shape) / self.prob
        out = X * self.mask
        return out.reshape(X.shape)

    def backward(self, dout):
        dX = dout * self.mask
        return dX, []


# In[ ]:




