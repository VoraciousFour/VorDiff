import numpy as np


class ReverseScalar():
    
    def __init__(self, val: float):
        
        self._val = val
        self._gradient = None
        self._children = {}
        
    def get(self):
        
        return self._val
    
    def compute_gradient(self):
        
        try:
            return sum([val * node.compute_gradient() for node, val in self.children.items()])
        except TypeError:
            return self._gradient
    
    def __add__(self, other):

        try: # If scalar
            child = ReverseScalar(self._val+other._val)
            self._children[child] = 1
            other._children[child] = 1
            return child
        
        except AttributeError: # If constant
            child = ReverseScalar(self._val+other)
            self._children[child] = 1
            return child
        
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
 
        try: # If scalar
            child = ReverseScalar(self._val-other._val)
            self._children[child] = 1
            other._children[child] = -1
            return child
       
        except AttributeError: # If constant
            child = ReverseScalar(self._val-other)
            self._children[child] = 1
            return child
        
    def __rsub__(self, other):
        
        child = ReverseScalar(other - self._val)
        self._children[child] = -1
        return child
    
    def __mul__(self, other):
        
        try: # If scalar
            child = ReverseScalar(self._val*other._val)
            self._children[child] = other._val
            other._children[child] = other._val
            return child
        
        except AttributeError: # If constant
            child = ReverseScalar(self._val*other)
            self._children[child] = other
            return child
        
    def __rmul__(self, other):
        
        return self.__mul__(other)
    
    def __truediv__(self, other):
        
        try: # If scalar
            child = ReverseScalar(self._val/other._val)
            self._children[child] = 1/other._val
            other._children[child] = -self._val/other._val**2
            return child
        
        except AttributeError: # If constant
            child = ReverseScalar(self._val/other)
            self._children[child] = 1/other
            return child
        
    def __rtruediv__(self, other):
        
        # Might be wrong
        
        try: # If scalar
            child = ReverseScalar(other._val/self._val)
            self._children[child] = 1/self._val
            other._children[child] = -other._val/self._val**2
            return child
        
        except AttributeError: # If constant
            child = ReverseScalar(other/self._val)
            self._children[child] = -other/self.val**2
            return child
        
    def __pow__(self, other):
        
        try: # If scalar
            child = ReverseScalar(self._val**other._val)
            self._children[child] = other._val*self._val**(other._val-1)
            other._children[child] = np.log(self._val)*self._val**other._val
            return child
        
        except AttributeError: # If constant
            child = ReverseScalar(self._val/other)
            self._children[child] = other*self._val**(other-1)
            return child
        
    def __rpow__(self, other):
        
        child = ReverseScalar(other**self._val)
        self._children[child] = np.log(other)*other**self._val
        return child
    
    def __neg__(self):
        
        child = ReverseScalar(-self._val)
        self.children[child] = -1
        return child
        
        