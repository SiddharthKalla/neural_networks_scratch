class Relu_Module:
  def __init__(self):
    print('Relu activation function constructed')
  def forward_pass(self,x):
    y = x * (x > 0)
    return y
  
  def backward_pass(self,dL_dy,y):
    dL_dx = dL_dy *( y > 0 )
    return dL_dx
