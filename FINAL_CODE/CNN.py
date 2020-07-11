#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[626]:


class CNN:
    
    def __init__(self , n_f , f = 3 , stride = 1 , padding = 'same',ep = 0):
        
        self.filter = None
        self.b = None
        self.padding = padding
        self.stride = stride
        self.n_f = n_f
        self.f = f
        self.ep = ep
    
    def Forward_pass(self,input_X):
        
        padding = self.padding
        stride = self.stride
        n_f = self.n_f                                            # No of filters
        f = self.f                                                # size of Filter 
        ep = self.ep
        
        (m,h,w,c) = input_X.shape                                 # size = ( m , h , w , c)
        if(ep==0):
            filt = np.random.randn(f,f,c,n_f)                     # size = ( f , f , c , n_f)
            b = np.random.randn(1,1,1,n_f)
            self.filter = filt
            self.b = b
        else:
            filt = self.filter
            b = self.b
        
        if(padding == 'same'):
            p = int((h*(stride - 1) - stride + f)/2)

            input_X = np.pad(input_X , ((0,0),(p,p),(p,p),(0,0)) , 'constant', constant_values=0)
        else: 
            p = 0

        o_h = int((h-f+2*p)/stride) + 1
        o_w = int((w-f+2*p)/stride) + 1

        output_X = np.zeros((m,o_h,o_w,n_f))
        
        for i in range(m):
            img = input_X[i]
            
            for height in range(o_h):
                for width in range(o_w):
                    for filter_no in range(n_f):
                        
                        img_ = img[(height*stride):(height*stride)+f,(width*stride):(width*stride)+f,:]
                        c_filt = filt[:,:,:,filter_no]
                        c_b = b[:,:,:,filter_no]
                        output_X[i,height,width,filter_no] = np.sum(np.multiply(img_,c_filt)) + float(c_b)
        
        print("Input shape = {} Filter shape = {} Output shape = {}".format((m,h,w,c),filt.shape,output_X.shape))

        self.ep = self.ep + 1
        return output_X
    
    
    def Backward_pass(self,input_X,dl_do,lr = 0.001):
        
        filt = self.filter
        b = self.b
        padding = self.padding
        stride = self.stride
        n_f = self.n_f 
        f = self.f
        
        (m,h,w,c) = input_X.shape                                 # size = ( m , h , w , c)
        
        dl_df = np.zeros(filt.shape)
        dl_dx = np.zeros(input_X.shape)
        db = np.zeros(b.shape)
        
        if(self.padding == 'same'):
            p = int((h*(stride - 1) - stride + f)/2)

            input_X_pad = np.pad(input_X , ((0,0),(p,p),(p,p),(0,0)))
            dl_dx_pad = np.pad(dl_dx , ((0,0),(p,p),(p,p),(0,0)))
        else: 
            p=0
            input_X_pad = input_X
            dl_dx_pad = dl_dx
            
        o_h = int((h-f+2*p)/stride) + 1
        o_w = int((w-f+2*p)/stride) + 1
        
        for i in range(m):
            img = input_X_pad[i]
            dl_dx_img = dl_dx_pad[i]
            
            for height in range(o_h):
                for width in range(o_w):
                    for filter_no in range(n_f):
                        
                        img_ = img[(height*stride):(height*stride)+f,(width*stride):(width*stride)+f,:]
                        
                        dl_dx_img[(height*stride):(height*stride)+f,(width*stride):(width*stride)+f,:] += filt[:,:,:,filter_no] * dl_do[i, height, width, filter_no]
                        dl_df[:,:,:,filter_no] += img_ * dl_do[i, height, width, filter_no]
                        db[:,:,:,filter_no] += dl_do[i, height, width, filter_no]
                        
            if(p!=0): dl_dx[i, :, :, :] = dl_dx_img[p:-p, p:-p, :]
            else: dl_dx[i, :, :, :] = dl_dx_img[:, :, :]
            
            filt = filt - lr*dl_df
            b = b - lr*db
            
            self.filter = filt
            self.b = b
            
        print("Backprop input = {}  Backprop output = {}".format(dl_do.shape,dl_dx.shape))
        return dl_dx
    


# In[ ]:




