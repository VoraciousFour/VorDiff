import numpy as np
from nodes.scalar import Scalar


class Operator:
    
    @staticmethod
    def sin(x):
        '''
        Returns the sine of a given constant or scalar
        
        INPUTS
        =======
        x: Numeric constant or Scalar object

        RETURNS
        =======
        sine of numeric constant or Scalar object
        
        NOTES
        =======
        If x is a constant, this method returns the constant sin(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative.
        '''
        
        try: # If scalar variable
            der = x._der 
            return Scalar(np.sin(x._val, np.cos(x._der)))
            
        except AttributeError: # If contant
            return np.sin(x)
        
    @staticmethod
    def cos(x):
        '''
        Returns the cosine of a given constant or scalar
        
        INPUTS
        =======
        x: Numeric constant or Scalar object

        RETURNS
        =======
        cosine of numeric constant or Scalar object
        
        NOTES
        =======
        If x is a constant, this method returns the constant cos(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative.
        '''
        
        try: # If scalar variable
            der = x._der 
            return Scalar(np.cos(x._val, -np.sin(x._der)))
            
        except AttributeError: # If contant
            return np.cos(x)
        
    @staticmethod
    def tan(x):
        '''
        Returns the tangent of a given constant or scalar
        
        INPUTS
        =======
        x: Numeric constant or Scalar object

        RETURNS
        =======
        tangent of numeric constant or Scalar object
        
        NOTES
        =======
        If x is a constant, this method returns the constant tan(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative.
        '''
        
        try: # If scalar variable
            der = x._der 
            return Scalar(np.tan(x._val, 1/np.cos(x._der)**2))
            
        except AttributeError: # If contant
            return np.tan(x)
        
    @staticmethod
    def arcsin(x):
        '''
        Returns the arcsine of a given constant or scalar
        
        INPUTS
        =======
        x: Numeric constant or Scalar object

        RETURNS
        =======
        arcsine of numeric constant or Scalar object
        
        NOTES
        =======
        If x is a constant, this method returns the constant arcsin(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative.
        '''
        
        try: # If scalar variable
            der = x._der 
            return Scalar(np.arcsin(x._val), 1/(1-x._der**2)**.5)
            
        except AttributeError: # If contant
            return np.arcsin(x)
        
    @staticmethod
    def arccos(x):
        '''
        Returns the arccosine of a given constant or scalar
        
        INPUTS
        =======
        x: Numeric constant or Scalar object

        RETURNS
        =======
        arccosine of numeric constant or Scalar object
        
        NOTES
        =======
        If x is a constant, this method returns the constant arccos(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative.
        '''
        
        try: # If scalar variable
            der = x._der 
            return Scalar(np.arccos(x._val), -1/(1-x._der**2)**.5)
            
        except AttributeError: # If contant
            return np.arcsin(x)
        
    @staticmethod
    def arctan(x):
        '''
        Returns the arctangent of a given constant or scalar
        
        INPUTS
        =======
        x: Numeric constant or Scalar object

        RETURNS
        =======
        arctangent of numeric constant or Scalar object
        
        NOTES
        =======
        If x is a constant, this method returns the constant arctan(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative.
        '''
        
        try: # If scalar variable
            der = x._der 
            return Scalar(np.arctan(x._val), 1/(1+x._der**2))
            
        except: # If contant
            return np.arctan(x)
        
    @staticmethod
    def log(x):
        '''
        Returns the log of a given constant or scalar
        
        INPUTS
        =======
        x: Numeric constant or Scalar object

        RETURNS
        =======
        log of numeric constant or Scalar object
        
        NOTES
        =======
        If x is a constant, this method returns the constant log(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative.
        '''
        
        try: # If scalar variable
            der = x._der 
            return Scalar(np.log(x._val), 1/x)
            
        except AttributeError: # If contant
            return np.log(x)
        
    @staticmethod
    def exp(x):
        '''
        Returns the exponential of a given constant or scalar
        
        INPUTS
        =======
        x: Numeric constant or Scalar object

        RETURNS
        =======
        exponential of numeric constant or Scalar object
        
        NOTES
        =======
        If x is a constant, this method returns the constant e^(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative.
        '''
        
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
      

