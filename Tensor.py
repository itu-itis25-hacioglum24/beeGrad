import numpy as np


class Teensor():
    
    def __init__(self,data,children=(),_op=''):
        
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data)
            
        self._backward = lambda:None
        self.op = _op
        self.prev = children
        self.grad = np.zeros_like(self.data,dtype=float)
        
    def __repr__(self):
        return f'Teensor(Data:{self.data},dtype:Teensor)'


        
    # ---- BASIC OPERATIONS -----

    def __add__(self,other):
        
        other = other if isinstance(other,Teensor) else Teensor(other)
        
        # Precaution for Numpy Broadcasting
        assert other.data.shape == self.data.shape, "Shape Mismatch!"
        
        out = Teensor(self.data + other.data, (self,other));out.op='+'
            
        ### (d(A+B)/dA = I[identity matrix])I  * out.grad (chain rule)
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
                
        out._backward = _backward
            
            
        return out
        
        
        
    def __mul__(self,other): # This operation is the Hadamard product, not matrix multiplication
        
        other = other if isinstance(other,Teensor) else Teensor(other)
        
        # Precaution for Numpy Broadcasting
        if self.data.shape != other.data.shape:
            assert self.data.size == 1 or other.data.size == 1, "Shape Mismatch!"
        
        out = Teensor(self.data * other.data, (self,other));out.op = '*'
            
        ### (d(A*B)/dA = B) B * out.grad(chain rule) 
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
                
        out._backward = _backward
                
        return out

    def __pow__(self,other): # This works via the Hadamard product , not matrix multiplication
        
        assert isinstance(other,(int,float)), "Only int and float are allowed for exponent"
        
        out = Teensor(self.data**other, (self, ));out.op = 'pow'
            
        def _backward():
            self.grad += other * (self.data**(other-1)) * out.grad
                
        out._backward = _backward
            
        return out
    
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1
    
        
    # ---- LINEAR  ALGEBRA OPERATIONS ----
    
    @property
    def transpose(self):
        out = Teensor(self.data.T,(self, ))
        # gradient of A^t = out.grad^t
        def _backward():
            self.grad = out.grad.T
        
        out._backward = _backward
            
        return out
    
    
    # Matrix Multiplication A@B
    def matmul(self,other):
        
        other = other if isinstance(other,Teensor) else Teensor(other)
        out = Teensor(self.data@other.data, (self,other), '@')
        
        def _backward():
            # upstream grad
            G = out.grad
            
            self.grad += G @ other.data.T
            other.grad += self.data.T @ G
        
        out._backward = _backward
        
        return out
    
    @property
    def inverse(self):
        out = Teensor(np.linalg.inv(self.data),(self, ),'inverse')
        
        def _backward():
            # upstream gradient
            G = out.grad
            self.grad += -out.data @ G @ out.data  # matmul
        out._backward = _backward
        
        return out
    
    def sum(self, axis=None, keepdims=False):
        
        data = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Teensor(data, (self,), 'sum')

        def _backward():
            grad = out.grad

            if axis is None:
                self.grad += grad * np.ones_like(self.data)

            else:
                expanded_grad = grad
                if not keepdims:
                    if isinstance(axis, int):
                        expanded_grad = np.expand_dims(grad, axis)
                    else:
                        for ax in sorted(axis):
                            expanded_grad = np.expand_dims(expanded_grad, ax)

                self.grad += expanded_grad * np.ones_like(self.data)

        out._backward = _backward
        
        return out

    # Compute gradients for all tensors
    def backward(self, grad=None):
            if grad is None:
                self.grad = np.ones_like(self.data)
            else:
                self.grad = grad

            visited = set()
            topo = []

            def build_topo(t):
                if t not in visited:
                    visited.add(t)
                    for child in t.prev:
                        build_topo(child)
                    topo.append(t)

            build_topo(self)

            for t in reversed(topo):
                t._backward()
