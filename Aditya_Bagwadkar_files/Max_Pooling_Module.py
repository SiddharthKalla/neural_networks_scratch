import numpy as np

class Max_Pooling_Module:
  def __init__(self,pool_size):
      print('Max Pooling network constructed')
      self.ps = pool_size
      
  def forward_pass(self,x):
    (samples,cx,hx,wx) = x.shape
    hy = int(hx/self.ps[0])
    wy = int(wx/self.ps[1])
    y = np.zeros([samples,cx,hy,wy])
    x_reshaped = np.reshape(x,(samples * cx * hx, wx))
    
    
    for j1 in range(self.ps[0]):
      for j2 in range(self.ps[1]):
        tmp = x_reshaped[j1::self.ps[0],j2::self.ps[1]]
        col = np.reshape(tmp,(-1,1))
        if (j1==0 and j2==0):
          x_col = col
        else:
          x_col = np.concatenate((x_col,col),axis=1)
        
    max_idx = np.argmax(x_col, axis=1)
    y = x_col[range(len(max_idx)),max_idx]
    y = np.reshape(y,(samples,cx,hy,-1))
    return y,max_idx

    
  def backward_pass(self,dL_dy,max_idx):
    (samples,cy,hy,wy) = dL_dy.shape
    dy_reshaped = np.reshape(dL_dy,(-1,1))
    hx = int(hy*self.ps[0])
    wx = int(wy*self.ps[1])
    
    dy_reshaped = np.transpose(dy_reshaped,(1,0))
    dx_reshaped = np.zeros([self.ps[0]*self.ps[1],len(max_idx)])
    dx_reshaped[max_idx,range(len(max_idx))] = dy_reshaped 
    dx_reshaped = np.transpose(dx_reshaped,(1,0))
                                                    
    img = np.zeros([samples*cy*hx,wx])
    
    for j1 in range(self.ps[0]):
      for j2 in range(self.ps[1]):
        col = dx_reshaped[:,j1*self.ps[0]+j2]
        tmp = img[j1::self.ps[0],j2::self.ps[1]]
        img[j1::self.ps[0],j2::self.ps[1]] = np.reshape(col,(tmp.shape))
       
    dL_dx = np.reshape(img,(samples,cy,hx,wx))
    return dL_dx
