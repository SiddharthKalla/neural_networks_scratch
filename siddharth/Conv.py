# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 17:13:46 2020

@author: leno
"""
import numpy as np

class Conv_op:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.conv_filter = np.random.rand(num_filters, filter_size, filter_size)/(filter_size * filter_size)
        
    def image_region(self, image):
        height, width = image.shape
        self.image = image
        for j in range(height - self.filter_size + 1):
            for k in range(width - self.filter_size + 1):
                image_patch = image[j: (j + self.filter_size), k: (k + self.filter_size)]
                yield image_patch, j, k
                
    def forward_prop(self, image):
        height, width = image.shape
        conv_out = np.zeros((height - self.filter_size + 1, width - self.filter_size + 1, self.num_filters))
        for image_patch, i , j in self.image_region(image):
            conv_out[i, j] = np.sum(image_patch*self.conv_filter, axis  = (1,2))
        return conv_out
    
    def back_prop(self, dL_dout, learning_rate):
        dL_dF_params = np.zeros(self.conv_filter.shape)
        for image_patch, i, j in self.image_region(self.image):
            for k in range(self.num_filters):
                dL_dF_params[k] += image_patch * dL_dout[i, j, k]
        #filter params update        
        self.conv_filter -= learning_rate * dL_dF_params
        return dL_dF_params
