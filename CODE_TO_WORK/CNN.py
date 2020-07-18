#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


class CNN:
    
    def __init__(self , n_f , f = 3 , stride = 1 , padding = 'same',ep = 0):
        
        self.filt = None
        self.b = None
        self.padding = padding
        self.stride = stride
        self.n_f = n_f
        self.f = f
        self.ep = ep
        self.dummy = None
    
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
            self.filt = filt
            self.b = b
        else:
            filt = self.filt
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
        
        if(ep==0) : print("Input shape = {} Filter shape = {} Output shape = {}".format((m,h,w,c),filt.shape,output_X.shape))

        self.ep = self.ep + 1
        return output_X
    
    
    def Backward_pass(self,input_X,dl_do,lr = 0.001):
        
        filt = self.filt
        b = self.b
        padding = self.padding
        stride = self.stride
        n_f = self.n_f 
        f = self.f
#         ep = self.ep
#         beta1 = 0.9
#         beta2 = 0.999
#         epsi = 10**(-8)
        
#         if(ep==0):
#             vo = np.zeros(filt.shape)
#             mo = np.zeros(filt.shape)
#             vob = np.zeros(b.shape)
#             mob = np.zeros(b.shape)
#         else : (mo,vo,mob,vob) = self.dummy
        
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
            img = input_X_pad[i,:,:,:]
            dl_dx_img = dl_dx_pad[i,:,:,:]
            
            for height in range(o_h):
                for width in range(o_w):
                    for filter_no in range(n_f):
                    
                        img_ = img[(height*stride):(height*stride)+f,(width*stride):(width*stride)+f,:]
                        
                        dl_dx_img[(height*stride):(height*stride)+f,(width*stride):(width*stride)+f,:] += (filt[:,:,:,filter_no] * dl_do[i, height, width, filter_no])
                        dl_df[:,:,:,filter_no] += img_ * dl_do[i, height, width, filter_no]
                        db[:,:,:,filter_no] += dl_do[i, height, width, filter_no]
                        
            if(p!=0): dl_dx[i, :, :, :] = dl_dx_img[p:-p, p:-p, :]
            else: dl_dx[i, :, :, :] = dl_dx_img[:, :, :]
            
        
#         ep += 1
#         mo = (mo*beta1) + (1-beta1)*dl_df 
#         vo = (vo*beta2) + (1-beta2)*(dl_df**2)
#         mo_ = mo/(1 - beta1**ep)
#         vo_ = vo/(1-beta2**ep)
#         mob = (mob*beta1) + (1-beta1)*db 
#         vob = (vob*beta2) + (1-beta2)*(db**2)
#         mob_ = mob/(1 - beta1**ep)
#         vob_ = vob/(1-beta2**ep)
        
#         self.filt = self.filt - (lr*mo_)/(np.sqrt(vo_) + epsi)
#         self.b = self.b - (lr*mob_)/(np.sqrt(vob_) + epsi)
        
#         self.dummy = (mo,vo,mob,vob)
#         self.ep = ep    


        self.filt -= lr*dl_df
        self.b -= lr*db
        #print("Backprop input = {}  Backprop output = {}".format(dl_do.shape,dl_dx.shape))
        return dl_dx


# In[ ]:




