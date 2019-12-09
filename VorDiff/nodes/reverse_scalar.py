import numpy as np


class ReverseScalar():
    
    def __init__(self, val: float):
        
        self._val = val
        self._gradient = 1
        self._children = {}
        
    def get(self):
        
        return self._val
    
    def compute_gradient(self):
        
        if len(self._children.keys()) > 0:
            
            gradient = 0
            for node, val in self._children.items():
                gradient += val*node.compute_gradient()
            return gradient
            #return sum([val * node.compute_gradient() for node, val in self._children.items()])
            #except TypeError:
            #    return self._gradient
        else:
            return 1
    
    def __add__(self, other):

        try: # If scalar
            child = ReverseScalar(self._val+other._val)
            child._children[self] = 1
            child._children[other] = 1
            return child
        
        except AttributeError: # If constant
            child = ReverseScalar(self._val+other)
            child._children[self] = 1
            return child
        
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
 
        try: # If scalar
            child = ReverseScalar(self._val-other._val)
            child._children[self] = 1
            child._children[other] = -1
            return child
       
        except AttributeError: # If constant
            child = ReverseScalar(self._val-other)
            child._children[self] = 1
            return child
        
    def __rsub__(self, other):
        
        try: # If scalar
            child = ReverseScalar(other._val- self._val)
            child._children[self] = -1
            child._children[other] = 1
            return child
       
        except AttributeError: # If constant
            child = ReverseScalar(other-self._val)
            child._children[self] = -1
            return child
    
    def __mul__(self, other):
        
        try: # If scalar
            child = ReverseScalar(self._val*other._val)
            child._children[self] = other._val
            child._children[other] = self._val
            return child
        
        except AttributeError: # If constant
            child = ReverseScalar(self._val*other)
            child._children[self] = other
            return child
        
    def __rmul__(self, other):
        
        return self.__mul__(other)
    
    def __truediv__(self, other):
        
        try: # If scalar
            child = ReverseScalar(self._val/other._val)
            child._children[self] = 1/other._val
            child._children[other] = -self._val/other._val**2
            return child
        
        except AttributeError: # If constant
            child = ReverseScalar(self._val/other)
            child._children[self] = 1/other
            return child
        
    def __rtruediv__(self, other):
        
        # Might be wrong
        
        try: # If scalar
            child = ReverseScalar(other._val/self._val)
            child._children[self] = -other._val/self._val**2
            child._children[other] = 1/self._val
            return child
        
        except AttributeError: # If constant
            child = ReverseScalar(other/self._val)
            child._children[self] = -other/self.val**2
            return child
        
    def __pow__(self, other):
        
        try: # If scalar
            child = ReverseScalar(self._val**other._val)
            child._children[self] = other._val*self._val**(other._val-1)
            child._children[other] = np.log(self._val)*self._val**other._val
            return child
        
        except AttributeError: # If constant
            child = ReverseScalar(self._val**other)
            child._children[self] = other*self._val**(other-1)
            return child
        
    def __rpow__(self, other):
        

        child = ReverseScalar(other**self._val)
        child._children[self] = np.log(self._val)*self._val**other
        return child
    
    def __neg__(self):
        
        child = ReverseScalar(-self._val)
        child.children[self] = -1
        return child
        
        