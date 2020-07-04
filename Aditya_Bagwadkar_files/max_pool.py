#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[42]:


class max_pool:
    
    def __init__(self,f,mode = "max",stride=1):
        
        self.mode = mode
        self.stride = stride
        self.f = f
        self.X = None
        
    def Forward_pass(self,input_X):
        
        mode = self.mode
        stride = self.stride
        self.X = input_X
        f = self.f
        
        (m,h,w,c) = input_X.shape                                 # size = ( m , h , w , c)
        
        o_h = int((h-f)/stride) + 1
        o_w = int((w-f)/stride) + 1
        
        output_X = np.zeros((m,o_h,o_w,c))
        
        for i in range(m):
            for height in range(o_h):
                for width in range(o_w):
                    for channel in range(c):
                        
                        img = input_X[i,(height*stride):(height*stride)+f,(width*stride):(width*stride)+f,channel]
                        if mode == "max":
                            output_X[i, height, width, channel] = np.max(img)
                        elif mode == "average":
                            output_X[i, height, width, channel] = np.mean(img)
        
        print("Input shape = {} Output shape = {}".format(self.X.shape,output_X.shape))
        return output_X
    
    def Backward_pass(self,dl_do):
        
        mode = self.mode
        stride = self.stride
        input_X = self.X
        f = self.f
        
        (m,h,w,c) = input_X.shape                                 # size = ( m , h , w , c)
        (m, o_h, o_w, c1) = dl_do.shape
        
        dx = np.zeros(input_X.shape)
        
        for i in range(m):
            img = input_X[i]
            
            for height in range(o_h):
                for width in range(o_w):
                    for channel in range(c1):
                                                
                        if mode == 'max':
                            
                            img_ = img[(height*stride):(height*stride)+f,(width*stride):(width*stride)+f,channel]
                            mask = img_ == np.max(img_)
                            dx[i,(height*stride):(height*stride)+f,(width*stride):(width*stride)+f, channel] += np.multiply(mask, dl_do[i, height, width, channel])
                            
                        elif mode == 'average':
                            
                            dx_ = dl_do[i, height, width, channel]
                            average = dx_ / (f*f)
                            a = np.ones((f,f)) * average
                            dx[i,(height*stride):(height*stride)+f,(width*stride):(width*stride)+f, channel] += a
        
        print("Backprop input = {}  Backprop output = {}".format(dl_do.shape,dx.shape))
        return dx
        


# In[ ]:

