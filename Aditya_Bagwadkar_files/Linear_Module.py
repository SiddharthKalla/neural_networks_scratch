import numpy as np
class Linear_Module:
  def __init__(self,shape_w, shape_b):
      print('Linear network constructed')
      range_w = np.sqrt(6. / (shape_w[0] + shape_w[1] + 1))
      range_b = np.sqrt(6. / (shape_b[0] + 1))
      self.w = np.random.uniform(-range_w,range_w,shape_w)
      self.b = np.random.uniform(-range_b,range_b,shape_b)
      
  def forward_pass(self,x):
    y = np.matmul(x,self.w)+self.b
    return y
  
  def backward_pass(self,dL_dy,x,lr):
    dL_dx = np.matmul(dL_dy,self.w.T)
    # SGD
    dw = np.matmul(x.T,dL_dy)
    db = np.sum(dL_dy,axis=0)
    self.b -= lr*db
    self.w -= lr*dw
    return dL_dx
