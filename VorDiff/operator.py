import numpy as np
from nodes.scalar import Scalar


class Operator:
    
    @staticmethod
    def sin(x):
        
        try: # If scalar variable
            der = x._der 
            return Scalar(np.sin(x._val, np.cos(x._der)))
            
        except AttributeError: # If contant
            return np.sin(x)
        
    @staticmethod
    def cos(x):
        
        try: # If scalar variable
            der = x._der 
            return Scalar(np.cos(x._val, -np.sin(x._der)))
            
        except AttributeError: # If contant
            return np.cos(x)
        
    @staticmethod
    def tan(x):
        
        try: # If scalar variable
            der = x._der 
            return Scalar(np.tan(x._val, 1/np.cos(x._der)**2))
            
        except AttributeError: # If contant
            return np.tan(x)
        
    @staticmethod
    def arcsin(x):
        
        try: # If scalar variable
            der = x._der 
            return Scalar(np.arcsin(x._val), 1/(1-x._der**2)**.5)
            
        except AttributeError: # If contant
            return np.arcsin(x)
        
    @staticmethod
    def arccos(x):
        
        try: # If scalar variable
            der = x._der 
            return Scalar(np.arccos(x._val), -1/(1-x._der**2)**.5)
            
        except AttributeError: # If contant
            return np.arcsin(x)
        
    @staticmethod
    def arctan(x):
        
        try: # If scalar variable
            der = x._der 
            return Scalar(np.arctan(x._val), 1/(1+x._der**2))
            
        except: # If contant
            return np.arctan(x)
        
    @staticmethod
    def log(x):
        
        try: # If scalar variable
            der = x._der 
            return Scalar(np.log(x._val), 1/x)
            
        except AttributeError: # If contant
            return np.log(x)
        
    @staticmethod
    def exp(x):
        
        try: # If scalar variable
            der = x._der 
            return Scalar(np.exp(x._val), np.exp(x))
            
        except AttributeError: # If contant
            return np.exp(x)
        
        