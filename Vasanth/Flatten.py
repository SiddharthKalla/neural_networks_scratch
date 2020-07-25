#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from im2col import *
class Flatten():

    def __init__(self):
        self.params = []

    def forward(self, X):
        self.X_shape = X.shape
        self.out_shape = (self.X_shape[0], -1)
        out = X.ravel().reshape(self.out_shape)
        self.out_shape = self.out_shape[1]
        return out

    def backward(self, dout):
        out = dout.reshape(self.X_shape)
        return out, ()


# In[ ]:




