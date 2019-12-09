import numpy as np
from VorDiff.nodes.reverse_scalar import ReverseScalar


class ReverseOperator:
    
    @staticmethod
    def sin(x):
        
        try: # If scalar variable
            child = ReverseScalar(np.sin(x._val))
            x._children[child] = np.cos(x._val)
            return child
        
        except AttributeError: # If contant
            return np.sin(x)
        
    @staticmethod
    def cos(x):
        
        try: # If scalar variable
            child = ReverseScalar(np.cos(x._val))
            x._children[child] = -np.sin(x._val)
            return child
        
        except AttributeError: # If contant
            return np.cos(x)
        
    @staticmethod
    def tan(x):
        
        try: # If scalar variable
            child = ReverseScalar(np.tan(x._val))
            x._children[child] = 1 + np.tan(x._val)**2
            return child
        
        except AttributeError: # If contant
            return np.tan(x)
        
    @staticmethod
    def sqrt(x):
        
        try: # If scalar variable
            child = ReverseScalar(np.sqrt(x._val))
            x._children[child] = .5/np.sqrt(x._val)
            return child
        
        except AttributeError: # If contant
            return np.sqrt(x)
        
    @staticmethod
    def log(x):
        
        try: # If scalar variable
            child = ReverseScalar(np.log(x._val))
            x._children[child] = 1/x._val
            return child
        
        except AttributeError: # If contant
            return np.log(x)
        
    @staticmethod
    def arcsin(x):
        
        try: # If scalar variable
            child = ReverseScalar(np.arcsin(x._val))
            x._children[child] = 1/np.sqrt(1-x._val**2)
            return child
        
        except AttributeError: # If contant
            return np.arcsin(x)
        
    @staticmethod
    def arccos(x):
        
        try: # If scalar variable
            child = ReverseScalar(np.arccos(x._val))
            x._children[child] = -1/np.sqrt(1-x._val**2)
            return child
        
        except AttributeError: # If contant
            return np.arccos(x)
        
    @staticmethod
    def arctan(x):
        
        try: # If scalar variable
            child = ReverseScalar(np.arctan(x._val))
            x._children[child] = 1/(1+x._val**2)
            return child
        
        except AttributeError: # If contant
            return np.arctan(x)
        
    @staticmethod
    def sinh(x):
        
        try: # If scalar variable
            child = ReverseScalar(np.sinh(x._val))
            x._children[child] = np.cosh(x._val)
            return child
        
        except AttributeError: # If contant
            return np.sinh(x)
        
    @staticmethod
    def cosh(x):
        
        try: # If scalar variable
            child = ReverseScalar(np.cosh(x._val))
            x._children[child] = np.sinh(x._val)
            return child
        
        except AttributeError: # If contant
            return np.cosh(x)
        
    @staticmethod
    def tanh(x):
        
        try: # If scalar variable
            child = ReverseScalar(np.tanh(x._val))
            x._children[child] = 1-np.tanh(x._val)**2
            return child
        
        except AttributeError: # If contant
            return np.tanh(x)
        
        