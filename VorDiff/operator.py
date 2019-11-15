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
        
    @staticmethod
    def sinh(x):
        try: # if scalar variable
            der = x._der
            return Scalar(np.sinh(x._val), np.cosh(x._val))
   
        except AttributeError: #if constant
            return np.sinh(x)  
      
    @staticmethod
    def cosh(x):
        try: # if scalar variable
            der = x._der
            return Scalar(np.cosh(x._val), np.sinh(x._val))
   
        except AttributeError: #if constant
            return np.cosh(x)

    @staticmethod
    def tanh(x):
        try: # if scalar variable
            der = x._der
            return Scalar(np.tanh(x._val), 1-np.tanh(x._val)**2)
   
        except AttributeError: #if constant
            return np.tanh(x)


    @staticmethod
    def arcsinh(x):
        try: # if scalar variable
            der = x._der
            return Scalar(np.arcsinh(x._val), -np.arcsinh(x._val)*np.arctanh(x._val))
   
        except AttributeError: #if constant
            return np.arcsinh(x)
        
        
        
    @staticmethod
    def arccosh(x):
        try: # if scalar variable
            der = x._der
            return Scalar(np.arccosh(x._val), -np.arccosh(x._val)*np.tanh(x._val))
   
        except AttributeError: #if constant
            return np.arccosh(x)
        
    @staticmethod
    def arctanh(x):
        try: # if scalar variable
            der = x._der
            return Scalar(np.arctanh(x._val), 1-np.arctanh(x._val)**2)
   
        except AttributeError: #if constant
            return np.arctanh(x)
      

