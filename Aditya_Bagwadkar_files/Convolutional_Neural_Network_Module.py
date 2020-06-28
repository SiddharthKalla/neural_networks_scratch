import numpy as np
import copy
class Convolutional_Neural_Network_Module:
  def __init__(self,shape_w, shape_b,strides=[1,1],padding='SAME'):
      print('Convolutional Neural Network constructed')
      range_w = np.sqrt(6. / (shape_w[0] + shape_w[1] +1))
      range_b = np.sqrt(6. / (shape_b[0] + 1))
      self.w = np.random.uniform(-range_w,range_w,shape_w)
      self.b = np.random.uniform(-range_b,range_b,shape_b)
      self.strides = strides
      self.padding = padding
    
  def forward_pass(self,x):
    F, C, HH, WW, = self.w.shape
    N, C, H, W = x.shape

    stride_h = self.strides[0]
    stride_w = self.strides[1]
    padding = self.padding
    
    if padding=='SAME':
      pad1 = int((HH-stride_h+H*(stride_h-1))/2.)
      pad2 = int((WW-stride_w+W*(stride_w-1))/2.)
    else:
      pad1,pad2 = 0,0
    
    Hy = int((H+2*pad1-HH)/stride_h)+1
    Wy = int((W+2*pad2-WW)/stride_w)+1
        
    
    padded = np.pad(x, [(0,0), (0,0), (pad1,pad1), (pad2,pad2)], 'constant')
      

    for j1 in range(C):
      for j2 in range(HH):
        for j3 in range(WW):
          tmp = padded[:,j1:j1+1,j2:(H+2*pad1+j2-HH+1):stride_h,j3:(W+2*pad2+j3-WW+1):stride_w]
          col = np.reshape(tmp,(-1,1))
          if (j1==0 and j2==0 and j3==0):
            x_col = col
          else:
            x_col = np.concatenate((x_col,col),axis=1)
     
    w_reshaped = np.reshape(self.w,[F,-1])
    w_reshaped = np.transpose(w_reshaped,(1,0))
    y = np.matmul(x_col,w_reshaped)+self.b
      
    y = np.reshape(y,(N,Hy,Wy,F))
    y = np.transpose(y,(0,3,1,2))

    return y
  
  def backward_pass(self,dL_dy,x,lr):
    F, C, HH, WW = self.w.shape
    N, C, H, W = x.shape
    
    stride_h = self.strides[0]
    stride_w = self.strides[1]
    padding = self.padding
    
    if padding=='SAME':
      pad1 = int((HH-stride_h+H*(stride_h-1))/2.)
      pad2 = int((WW-stride_w+W*(stride_w-1))/2.)
    else:
      pad1,pad2 = 0,0
    
    Hy = int((H+2*pad1-HH)/stride_h)+1
    Wy = int((W+2*pad2-WW)/stride_w)+1
    
    
    dx = np.zeros_like(x)
    dw = np.zeros_like(self.w)
    db = np.zeros_like(self.b)
    
    b = copy.deepcopy(self.b)
    w_reshaped = np.reshape(copy.deepcopy(self.w),[F,-1])
    
    padded = np.pad(x, [(0,0), (0,0), (pad1,pad1), (pad2,pad2)], 'constant')
    padded_dx = np.pad(dx, [(0,0), (0,0), (pad1,pad1), (pad2,pad2)], 'constant')
   

    for j1 in range(C):
      for j2 in range(HH):
        for j3 in range(WW):
          tmp = padded[:,j1:j1+1,j2:(H+2*pad1+j2-HH+1):stride_h,j3:(W+2*pad2+j3-WW+1):stride_w]
          col = np.reshape(tmp,(-1,1))
          if (j1==0 and j2==0 and j3==0):
            x_col = col
          else:
            x_col = np.concatenate((x_col,col),axis=1)
            
    y2 = np.transpose(dL_dy,(0,2,3,1))
    y2 = np.reshape(y2,(-1,F))
    
    dx_col = np.matmul(y2,w_reshaped)
      
    for j1 in range(C):
      for j2 in range(HH):
        for j3 in range(WW):
          col_ind = j1*HH*WW+j2*WW+j3
          col = dx_col[:,col_ind:col_ind+1]
          block = np.reshape(col,[-1,1,Hy,Wy])
          padded_dx[:,j1:j1+1,j2:(H+2*pad1+j2-HH+1):stride_h,j3:(W+2*pad2+j3-WW+1):stride_w] += block
         
    
    dx = padded_dx[:, :, pad1:pad1+H, pad2:pad2+W]
            
    dout = np.transpose(dL_dy,(0,2,3,1))
    dout = np.reshape(dout,(-1,F))
    
    w_reshaped -= lr * np.matmul(dout.T, x_col)
    b -= lr * np.sum(dout.T, axis=1)
  
    w = np.reshape(w_reshaped,[F,C,HH,WW])
    
    self.w = w
    self.b = b

    return dx
