# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 17:23:06 2020

@author: leno
"""
import numpy as np

class softmax:
    def __init__(self, input_node, softmax_node):
        self.weight = np.random.rand(input_node, softmax_node)/input_node
        self.bias = np.zeros(softmax_node)
        
    def forward_prop(self, image):
        self.orig_im_shape = image.shape
        image_modified = image.flatten()
        self.modified_input = image_modified
        output_val = np.dot(image_modified, self.weight) + self.bias
        self.out = output_val
        exp_out = np.exp(output_val)
        return exp_out/np.sum(exp_out, axis = 0)
    
    def back_prop(self, dL_dout, learning_rate):
        for i, grad in enumerate(dL_dout):
            if grad == 0:
                continue
            transformation_eq = np.exp(self.out)
            S_total = np.sum(transformation_eq)
            
            #gradient with respect to out(z)
            dy_dz = -transformation_eq[i]*transformation_eq/(S_total **2)
            dy_dz[i] = transformation_eq[i]*(S_total - transformation_eq[i])/(S_total **2)
            
            #gradients of totals against weights/bias/input
            dz_dw = self.modified_input
            dz_db = 1
            dz_d_inp = self.weight
            
            #gradients of loss against totals
            dL_dz = grad * dy_dz
            
            #gradients of loss against weights/bias/input
            dL_dw = dz_dw[np.newaxis].T @ dL_dz[np.newaxis]
            dL_db = dL_dz * dz_db
            dL_d_inp = dz_d_inp @ dL_dz
            
            #update weights and biases
            self.weight -= learning_rate * dL_dw
            self.bias -= learning_rate * dL_db
            
            return dL_d_inp.reshape(self.orig_im_shape)
