import numpy as np


class Teensor():
    
    def __init__(self,data,children=(),_op=''):
        self.data = np.array(data,dtype=float)
        self._backward = lambda:None
        self.op = _op
        self.prev = children
        self.grad = np.zeros_like(self.data,dtype=float)
        
        def __repr__(self):
            return f'Teensor(Data:{self.data},dtype:Teensor)'


        
        # ---- BASIC OPERATIONS -----

        def __add__(self,other):
            other = other if isinstance(other,Teensor) else Teensor(other)
            out = Teensor(self.data + other.data, (self,other));out.op='+'
            
            ### (d(A+B)/dA = I[identity matrix])I  * out.grad (chain rule)
            def _backward():
                self.grad += out.grad
                other.grad += out.grad
                for child in out.prev:
                    child._backward()
                
            out._backward = _backward
            
            
            return out
        
        
        
        def __mul__(self,other): # This operation is the Hadamard product, not matrix multiplication
            other = other if isinstance(other,Teensor) else Teensor(other)
            out = Teensor(self.data * other.data, (self,other));out.op = '*'
            
            ### (d(A*B)/dA = B) B * out.grad(chain rule) 
            def _backward():
                self.grad = other.data * out.grad
                other.grad = self.data * out.grad
                
            out._backward = _backward
                
            return out

        def __pow__(self,other): # This works via the Hadamard product , not matrix multiplication
            other = other if isinstance(other,Teensor) else Teensor(other)
            out = Teensor(self.data**other.data, (self,other));out.op = 'pow'
            
            def _backward():
                self.grad = other * (self.data**(other-1)) * out.grad
                
            out._backward = _backward
            
            return out
    
        def __sub__(self,other):
            other = other if isinstance(other,Teensor) else Teensor(other)
            out = Teensor(self.data+(-other.data));out.op = '-'
            return out
        
        def __truediv__(self,other):
            other = other if isinstance(other,Teensor) else Teensor(other)
            out = Teensor(self.data / other.data);out.op = '/'
            return out
        
        def __neg__(self):
            return Teensor(self.data*(-1))
        
        # ---- REVERSED BASIC OPERATIONS ----
        
        def __radd__(self,other):
            return Teensor(self.data + other)
        
        def __rsub__(self,other):
            return Teensor(other - self.data)
        
        def __rtruediv__(self,other):
            return Teensor(other / self.data)
        
        
    
        
        


    
      
