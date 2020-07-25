#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from im2col import *
class Batchnorm():

    def __init__(self, X_dim):
        self.d_X, self.h_X, self.w_X = X_dim
        self.gamma = np.ones((1, int(np.prod(X_dim))))
        self.beta = np.zeros((1, int(np.prod(X_dim))))
        self.params = [self.gamma, self.beta]

    def forward(self, X):
        self.n_X = X.shape[0]
        self.X_shape = X.shape

        self.X_flat = X.ravel().reshape(self.n_X, -1)
        self.mu = np.mean(self.X_flat, axis=0)
        self.var = np.var(self.X_flat, axis=0)
        self.X_norm = (self.X_flat - self.mu) / np.sqrt(self.var + 1e-8)
        out = self.gamma * self.X_norm + self.beta

        return out.reshape(self.X_shape)

    def backward(self, dout):

        dout = dout.ravel().reshape(dout.shape[0], -1)
        X_mu = self.X_flat - self.mu
        var_inv = 1. / np.sqrt(self.var + 1e-8)

        dbeta = np.sum(dout, axis=0)
        dgamma = np.sum(dout * self.X_norm, axis=0)

        dX_norm = dout * self.gamma
        dvar = np.sum(dX_norm * X_mu, axis=0) * -             0.5 * (self.var + 1e-8)**(-3 / 2)
        dmu = np.sum(dX_norm * -var_inv, axis=0) + dvar *             1 / self.n_X * np.sum(-2. * X_mu, axis=0)
        dX = (dX_norm * var_inv) + (dmu / self.n_X) +             (dvar * 2 / self.n_X * X_mu)

        dX = dX.reshape(self.X_shape)
        return dX, [dgamma, dbeta]


# In[ ]:




