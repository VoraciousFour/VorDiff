import numpy as np
from VorDiff.nodes.scalar import Scalar


class Operator():
    
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
            return Scalar(np.sin(x._val), x._der*np.cos(x._val))
            
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
            return Scalar(np.cos(x._val), -np.sin(x._val)*x._der)
            
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
            return Scalar(np.tan(x._val), x._der/np.cos(x._val)**2)
            
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
            return Scalar(np.arcsin(x._val), 1/(x._der*(1-x._val**2)**.5))
            
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
            return Scalar(np.arccos(x._val), -x._der/(1-x._val**2)**.5)
            
        except AttributeError: # If contant
            return np.arccos(x)
        
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
            return Scalar(np.arctan(x._val), x._der/(1+x._val**2))
            
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
            return Scalar(np.log(x._val), x._der/x._val)
            
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
            return Scalar(np.exp(x._val), x._der*np.exp(x._val))
            
        except AttributeError: # If contant
            return np.exp(x)
        
    @staticmethod
    def sinh(x):
        '''
        Returns the sinh of a given constant or scalar
        
        INPUTS
        =======
        x: Numeric constant or Scalar object

        RETURNS
        =======
        sinh of numeric constant or Scalar object
        
        NOTES
        =======
        If x is a constant, this method returns the constant sinh(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative.
        '''
        
        try: # if scalar variable
            return Scalar(np.sinh(x._val), x._der*(np.cosh(x._val)))
   
        except AttributeError: #if constant
            return np.sinh(x)  
      
    @staticmethod
    def cosh(x):
        '''
        Returns the cosh of a given constant or scalar
        
        INPUTS
        =======
        x: Numeric constant or Scalar object

        RETURNS
        =======
        cosh of numeric constant or Scalar object
        
        NOTES
        =======
        If x is a constant, this method returns the constant cosh(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative.
        '''
        try: # if scalar variable
            return Scalar(np.cosh(x._val), x._der*(np.sinh(x._val)))
   
        except AttributeError: #if constant
            return np.cosh(x)

    @staticmethod
    def tanh(x):
        '''
        Returns the tanh of a given constant or scalar
        
        INPUTS
        =======
        x: Numeric constant or Scalar object

        RETURNS
        =======
        tanh of numeric constant or Scalar object
        
        NOTES
        =======
        If x is a constant, this method returns the constant tanh(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative.
        '''
        try: # if scalar variable
            return Scalar(np.tanh(x._val), x._der*(1-np.tanh(x._val)**2))
   
        except AttributeError: #if constant
            return np.tanh(x)


    @staticmethod
    def arcsinh(x):
        '''
        Returns the arcsinh of a given constant or scalar
        
        INPUTS
        =======
        x: Numeric constant or Scalar object

        RETURNS
        =======
        arcsinh of numeric constant or Scalar object
        
        NOTES
        =======
        If x is a constant, this method returns the constant arcsinh(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative.
        '''
        try: # if scalar variable
            return Scalar(np.arcsinh(x._val), x._der*(-np.arcsinh(x._val)*np.arctanh(x._val)))
   
        except AttributeError: #if constant
            return np.arcsinh(x)
        
        
        
    @staticmethod
    def arccosh(x):
        '''
        Returns the arccosh of a given constant or scalar
        
        INPUTS
        =======
        x: Numeric constant or Scalar object

        RETURNS
        =======
        arccosh of numeric constant or Scalar object
        
        NOTES
        =======
        If x is a constant, this method returns the constant arccosh(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative.
        '''
        try: # if scalar variable
            return Scalar(np.arccosh(x._val), x._der*(-np.arccosh(x._val)*np.tanh(x._val)))
   
        except AttributeError: #if constant
            return np.arccosh(x)
        
    @staticmethod
    def arctanh(x):
        '''
        Returns the arctanh of a given constant or scalar
        
        INPUTS
        =======
        x: Numeric constant or Scalar object

        RETURNS
        =======
        arctanh of numeric constant or Scalar object
        
        NOTES
        =======
        If x is a constant, this method returns the constant arctanh(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative.
        '''
        try: # if scalar variable
            return Scalar(np.arctanh(x._val), x._der*(1-np.arctanh(x._val)**2))
   
        except AttributeError: #if constant
            return np.arctanh(x)
      

