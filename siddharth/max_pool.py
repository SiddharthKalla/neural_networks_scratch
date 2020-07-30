# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 17:21:20 2020

@author: leno
"""
import numpy as np

class max_pool:
    def __init__(self, filter_size):
        self.filter_size = filter_size
        
    def image_region(self, image):
        new_height = image.shape[0] // self.filter_size
        new_width  = image.shape[1] // self.filter_size
        self.image = image
        
        for i in range(new_height):
            for j in range(new_width):
                image_patch = image[(i*self.filter_size):(i*self.filter_size + self.filter_size), (j*self.filter_size):(j*self.filter_size + self.filter_size)]
                yield image_patch, i, j
                
    def forward_prop(self, image):
        height, width, num_filters = image.shape
        output = np.zeros((height // self.filter_size, width // self.filter_size, num_filters))
        
        for image_patch, i, j in self.image_region(image):
            output[i,j] = np.amax(image_patch, axis = (0,1))
        return output
    def back_prop(self, dL_dout):
        dL_dmax_pool = np.zeros(self.image.shape)
        for image_patch, i, j in self.image_region(self.image):
            height, width, num_filters = image_patch.shape
            maximum_val = np.amax(image_patch, axis = (0,1))
            
            for i1 in range(height):
                for j1 in range(width):
                    for k1 in range(num_filters):
                        if image_patch[i1,j1,k1] == maximum_val[k1]:
                            dL_dmax_pool[i*self.filter_size + i1, j*self.filter_size + j1, k1] = dL_dout[i,j,k1]
                            
            return dL_dmax_pool
