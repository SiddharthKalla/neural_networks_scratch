import numpy as np
class Softmax_Module:
  def __init__(self):
    print('Softmax activation function constructed')
  def forward_pass(self,x):
    y = np.exp(x - np.max(x, axis=1, keepdims=True))
    y /= np.sum(y, axis=1,keepdims=True)
    return y
