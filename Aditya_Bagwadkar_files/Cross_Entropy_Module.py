import numpy as np
class Cross_Entropy_Module:
  def __init__(self):
    print('Cross-entropy constructed')
  def forward_pass(self,x,y_):
    y= np.sum(np.log((np.sum(np.exp(x), axis=1)))-x[range(len(x)),np.argmax(y_,1)])
    return y
  
  def backward_pass(self,x,y_): 
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    dx = probs.copy()
    dx[range(len(x)),np.argmax(y_,1)] -= 1
    return dx
