#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[42]:


class max_pool:
    
    def __init__(self,f,mode = "max",stride=1,ep=0):
        
        self.mode = mode
        self.stride = stride
        self.f = f
        self.ep = ep
        
    def Forward_pass(self,input_X):
        
        mode = self.mode
        stride = self.stride
        f = self.f
        ep = self.ep
        
        (m,c,h,w) = input_X.shape                                 # size = ( m , h , w , c)
        
        o_h = int((h-f)/stride) + 1
        o_w = int((w-f)/stride) + 1
        
        output_X = np.zeros((m,c,o_h,o_w))
        
        for i in range(m):
            for channel in range(c):
                for height in range(o_h):
                    for width in range(o_w):
                        
                        img = input_X[i,channel,(height*stride):(height*stride)+f,(width*stride):(width*stride)+f]
                        if mode == "max":
                            output_X[i, channel, height, width] = np.max(img)
                        elif mode == "average":
                            output_X[i, channel, height, width] = np.mean(img)
        
        if(ep==0): print("Input shape = {} Output shape = {}".format(input_X.shape,output_X.shape))
            
        self.ep = self.ep + 1
        return output_X
    
    def Backward_pass(self,input_X,dl_do):
        
        mode = self.mode
        stride = self.stride
        f = self.f
        
        (m,c,h,w) = input_X.shape                                 # size = ( m , h , w , c)
        (m,c1, o_h, o_w) = dl_do.shape
        
        dx = np.zeros(input_X.shape)
        
        for i in range(m):
            img = input_X[i]
            
            for channel in range(c1):
                for height in range(o_h):
                    for width in range(o_w):
                                                
                        if mode == 'max':
                            
                            img_ = img[channel,(height*stride):(height*stride)+f,(width*stride):(width*stride)+f]
                            mask = img_ == np.max(img_)
                            dx[i,channel,(height*stride):(height*stride)+f,(width*stride):(width*stride)+f] += np.multiply(mask, dl_do[i,channel, height, width])
                            
                        elif mode == 'average':
                            
                            dx_ = dl_do[i,channel, height, width]
                            average = dx_ / (f*f)
                            a = np.ones((f,f)) * average
                            dx[i,channel,(height*stride):(height*stride)+f,(width*stride):(width*stride)+f] += a
        
        #print("Backprop input = {}  Backprop output = {}".format(dl_do.shape,dx.shape))
        return dx
        


# In[ ]:




