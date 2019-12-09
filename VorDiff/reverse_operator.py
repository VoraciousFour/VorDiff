import numpy as np
from VorDiff.nodes.reverse_scalar import ReverseScalar
from VorDiff.nodes.reverse_vector import ReverseVector


class ReverseOperator:
    
    @staticmethod
    def sin(x):
        try:
            x._reverse_scalars
            child = ReverseVector(np.sin(x._val))
            for var in x._children.keys():
                child._children[var] = [(x,np.cos(x._val))]
            return child
        except AttributeError:
            try: # If scalar variable
                child = ReverseScalar(np.sin(x._val))
                child._children[x] = np.cos(x._val)
                return child
            
            except AttributeError: # If contant
                return np.sin(x)
        
    @staticmethod
    def cos(x):        
        try:
            x._reverse_scalars
            child = ReverseVector(np.cos(x._val))
            for var in x._children.keys():
                child._children[var] = [(x,-np.sin(x._val))]
            return child
        except AttributeError:
            try: # If scalar variable
                child = ReverseScalar(np.cos(x._val))
                child._children[x] = -np.sin(x._val)
                return child
            
            except AttributeError: # If contant
                return np.cos(x)
        
    @staticmethod
    def tan(x):
        try:
            x._reverse_scalars
            child = ReverseVector(np.tan(x._val))
            for var in x._children.keys():
                child._children[var] = [(x,1 + np.tan(x._val)**2)]
            return child
        except AttributeError:
            try: # If scalar variable
                child = ReverseScalar(np.tan(x._val))
                child._children[x] = 1 + np.tan(x._val)**2
                return child
            
            except AttributeError: # If contant
                return np.tan(x)        
 
    @staticmethod
    def sqrt(x):
        try:
            x._reverse_scalars
            child = ReverseVector(np.sqrt(x._val))
            for var in x._children.keys():
                child._children[var] = [(x,.5/np.sqrt(x._val))]
            return child
        except AttributeError:
            try: # If scalar variable
                child = ReverseScalar(np.sqrt(x._val))
                child._children[x] = .5/np.sqrt(x._val)
                return child
            
            except AttributeError: # If contant
                return np.sqrt(x)         
        
    @staticmethod
    def log(x):
        try:
            x._reverse_scalars
            child = ReverseVector(np.log(x._val))
            for var in x._children.keys():
                child._children[var] = [(x,1/x._val)]
            return child
        except AttributeError:
            try: # If scalar variable
                child = ReverseScalar(np.log(x._val))
                child._children[x] = 1/x._val
                return child
            
            except AttributeError: # If contant
                return np.log(x)         

    @staticmethod
    def arcsin(x):
        try:
            x._reverse_scalars
            child = ReverseVector(np.arcsin(x._val))
            for var in x._children.keys():
                child._children[var] = [(x,1/np.sqrt(1-x._val**2))]
            return child
        except AttributeError:
            try: # If scalar variable
                child = ReverseScalar(np.arcsin(x._val))
                child._children[x] = 1/np.sqrt(1-x._val**2)
                return child
            
            except AttributeError: # If contant
                return np.arcsin(x)       

        
    @staticmethod
    def arccos(x):
        try:
            x._reverse_scalars
            child = ReverseVector(np.arccos(x._val))
            for var in x._children.keys():
                child._children[var] = [(x,-1/np.sqrt(1-x._val**2))]
            return child
        except AttributeError:
            try: # If scalar variable
                child = ReverseScalar(np.arccos(x._val))
                child._children[x] = -1/np.sqrt(1-x._val**2)
                return child
            
            except AttributeError: # If contant
                return np.arccos(x)         
        
    @staticmethod
    def arctan(x):
        try:
            x._reverse_scalars
            child = ReverseVector(np.arctan(x._val))
            for var in x._children.keys():
                child._children[var] = [(x,1/(1+x._val**2))]
            return child
        except AttributeError:
            try: # If scalar variable
                child = ReverseScalar(np.arctan(x._val))
                child._children[x] = 1/(1+x._val**2)
                return child
            
            except AttributeError: # If contant
                return np.arctan(x)              
        
    @staticmethod
    def sinh(x):
        try:
            x._reverse_scalars
            child = ReverseVector(np.sinh(x._val))
            for var in x._children.keys():
                child._children[var] = [(x,np.cosh(x._val))]
            return child
        except AttributeError:
            try: # If scalar variable
                child = ReverseScalar(np.sinh(x._val))
                child._children[x] = np.cosh(x._val)
                return child
            
            except AttributeError: # If contant
                return np.sinh(x)          

    @staticmethod
    def cosh(x):
        try:
            x._reverse_scalars
            child = ReverseVector(np.cosh(x._val))
            for var in x._children.keys():
                child._children[var] = [(x,np.sinh(x._val))]
            return child
        except AttributeError:
            try: # If scalar variable
                child = ReverseScalar(np.cosh(x._val))
                child._children[x] = np.sinh(x._val)
                return child
            
            except AttributeError: # If contant
                return np.cosh(x)         

        
    @staticmethod
    def tanh(x):
        try:
            x._reverse_scalars
            child = ReverseVector(np.tanh(x._val))
            for var in x._children.keys():
                child._children[var] = [(x,1-np.tanh(x._val)**2)]
            return child
        except AttributeError:
            try: # If scalar variable
                child = ReverseScalar(np.tanh(x._val))
                child._children[x] = 1-np.tanh(x._val)**2
                return child
            
            except AttributeError: # If contant
                return np.tanh(x)        

        
        
        
        
        
        
        
        