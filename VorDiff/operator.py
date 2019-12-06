import numpy as np
from VorDiff.nodes.scalar import Scalar
from VorDiff.nodes.vector import Vector


class Operator():
    
    @staticmethod
    def sin(x):
        '''
        Returns the sine of a given constant, Scalar, or Vector object
        
        INPUTS
        =======
        x: Numeric constant, Scalar object or Vector object

        RETURNS
        =======
        sine of numeric constant, Scalar object or Vector object
        
        NOTES
        =======
        If x is a constant, this method returns the constant sin(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative. If x is a Vector object, this method returns a new 
        Vector with the appropriate functions performed on the Vector's 
        value and Jacobian matrix.
        '''
        
        try:
            return Vector(np.sin(x._val), np.cos(x._val)*x._jacob)
        except AttributeError: # If contant
            try: # If scalar variable
                return Scalar(np.sin(x._val), x._der*np.cos(x._val))
            
            except AttributeError: # If contant
                return np.sin(x)
        
    @staticmethod
    def cos(x):
        '''
        Returns the cosine of a given constant, scalar or vector
        
        INPUTS
        =======
        x: Numeric constant, Scalar object or Vector object

        RETURNS
        =======
        cosine of numeric constant or Scalar object or Vector object
        
        NOTES
        =======
        If x is a constant, this method returns the constant sin(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative. If x is a Vector object, this method returns a new 
        Vector with the appropriate functions performed on the Vector's 
        value and Jacobian matrix.
        '''
        
        try:
            return Vector(np.cos(x._val), -np.sin(x._val)*x._jacob)
        except AttributeError: # If contant
            try: # If scalar variable
                return Scalar(np.cos(x._val), -np.sin(x._val)*x._der)
            
            except AttributeError: # If contant
                return np.cos(x)
        
    @staticmethod
    def tan(x):
        '''
        Returns the tangent of a given constant or scalar or Vector object
        
        INPUTS
        =======
        x: Numeric constant or Scalar object or Vector object

        RETURNS
        =======
        tangent of numeric constant or Scalar object or Vector object
        
        NOTES
        =======
        If x is a constant, this method returns the constant sin(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative. If x is a Vector object, this method returns a new 
        Vector with the appropriate functions performed on the Vector's 
        value and Jacobian matrix.
        ''' 
        try:
            return Vector(np.tan(x._val), x._jacob/np.cos(x._val)**2)
        except AttributeError: # If contant
            try: # If scalar variable
                return Scalar(np.tan(x._val), x._der/np.cos(x._val)**2)
            
            except AttributeError: # If contant
                return np.tan(x)
        
    @staticmethod
    def arcsin(x):
        '''
        Returns the arcsine of a given constant or scalar or Vector object
        
        INPUTS
        =======
        x: Numeric constant or Scalar object or Vector object
 
        RETURNS
        =======
        arcsine of numeric constant or Scalar object or Vector object
        
        NOTES
        =======
        If x is a constant, this method returns the constant sin(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative. If x is a Vector object, this method returns a new 
        Vector with the appropriate functions performed on the Vector's 
        value and Jacobian matrix.
        '''
        
        try:
            j = x._jacob
            if x._val<-1 or x._val>1:
                raise ValueError('out of domain')
            else:
                return Vector(np.arcsin(x._val), 1/(x._jacob*(1-x._val**2)**.5))
        except AttributeError:
        try: # If scalar variable
            if x._val<-1 or x._val>1:
                raise ValueError('out of domain')
            else:
                return Scalar(np.arcsin(x._val), 1/(x._der*(1-x._val**2)**.5))
            
        except AttributeError: # If contant
            if x<-1 or x>1:
                raise ValueError('out of domain')
            else:
                return np.arcsin(x)
        
    @staticmethod
    def arccos(x):
        '''
        Returns the arccosine of a given constant or scalar or Vector object
        
        INPUTS
        =======
        x: Numeric constant or Scalar object or Vector object

        RETURNS
        =======
        arccosine of numeric constant or Scalar object or Vector object
        
        NOTES
        =======
        If x is a constant, this method returns the constant sin(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative. If x is a Vector object, this method returns a new 
        Vector with the appropriate functions performed on the Vector's 
        value and Jacobian matrix.
        '''
        
        try:
            j = x._jacob
            if x._val<-1 or x._val>1:
                raise ValueError('out of domain')
            else:
                return Vector(np.arccos(x._val), -x._jacob/(1-x._val**2)**.5)
        except AttributeError:
            try: # If scalar variable
                if x._val<-1 or x._val>1:
                    raise ValueError('out of domain')
                else:
                    return Scalar(np.arccos(x._val), -x._der/(1-x._val**2)**.5)
            
            except AttributeError: # If contant
                if x<-1 or x>1:
                    raise ValueError('out of domain')
                else:
                    return np.arccos(x)
        
    @staticmethod
    def arctan(x):
        '''
        Returns the arctangent of a given constant or scalar or Vector object
        
        INPUTS
        =======
        x: Numeric constant or Scalar object or Vector object

        RETURNS
        =======
        arctangent of numeric constant or Scalar object or Vector object
        
        NOTES
        =======
        If x is a constant, this method returns the constant sin(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative. If x is a Vector object, this method returns a new 
        Vector with the appropriate functions performed on the Vector's 
        value and Jacobian matrix.
        '''
        
        try:
            return Vector(np.arctan(x._val), x._jacob/(1+x._val**2))
        except AttributeError:
            try: # If scalar variable
                return Scalar(np.arctan(x._val), x._der/(1+x._val**2))
            
            except: # If contant
                return np.arctan(x)
        
    @staticmethod
    def log(x):
        '''
        Returns the log of a given constant or scalar or Vector object
        
        INPUTS
        =======
        x: Numeric constant or Scalar object or Vector object

        RETURNS
        =======
        log of numeric constant or Scalar object or Vector object
        
        NOTES
        =======
        If x is a constant, this method returns the constant sin(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative. If x is a Vector object, this method returns a new 
        Vector with the appropriate functions performed on the Vector's 
        value and Jacobian matrix.
        '''
        
        try:
            j = x._jacob
            if x._val<=0:
                raise ValueError('out of domain')
            else:
                return Vector(np.log(x._val), x._jacob/x._val)
        except AttributeError:
            try: # If scalar variable
                if x._val<=0:
                    raise ValueError('out of domain')
                else:    
                    return Scalar(np.log(x._val), x._der/x._val)
            
            except AttributeError: # If contant
                if x<=0:
                    raise ValueError('out of domain')
                else:
                    return np.log(x)
        
    @staticmethod
    def exp(x):
        '''
        Returns the exponential of a given constant or scalar or Vector object
        
        INPUTS
        =======
        x: Numeric constant or Scalar object or Vector object

        RETURNS
        =======
        exponential of numeric constant or Scalar object or Vector object
        
        NOTES
        =======
        If x is a constant, this method returns the constant sin(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative. If x is a Vector object, this method returns a new 
        Vector with the appropriate functions performed on the Vector's 
        value and Jacobian matrix.
        '''
        
        try:
            return Vector(np.exp(x._val), x._jacob*(np.exp(x._val)))
        except AttributeError:
            try: # If scalar variable
                return Scalar(np.exp(x._val), x._der*np.exp(x._val))
            
            except AttributeError: # If contant
                return np.exp(x)
        
    @staticmethod
    def sinh(x):
        '''
        Returns the sinh of a given constant or scalar or Vector object
        
        INPUTS
        =======
        x: Numeric constant or Scalar object or Vector object

        RETURNS
        =======
        sinh of numeric constant or Scalar object or Vector object
        
        NOTES
        =======
        If x is a constant, this method returns the constant sin(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative. If x is a Vector object, this method returns a new 
        Vector with the appropriate functions performed on the Vector's 
        value and Jacobian matrix.
        '''
        try:
            return Vector(np.sinh(x._val), x._jacob*(np.cosh(x._val)))
        except AttributeError:
        
            try: # if scalar variable
                return Scalar(np.sinh(x._val), x._der*(np.cosh(x._val)))
   
            except AttributeError: #if constant
                return np.sinh(x)  
      
    @staticmethod
    def cosh(x):
        '''
        Returns the cosh of a given constant or scalar or Vector object
        
        INPUTS
        =======
        x: Numeric constant or Scalar object or Vector object

        RETURNS
        =======
        cosh of numeric constant or Scalar object or Vector object
        
        NOTES
        =======
        If x is a constant, this method returns the constant sin(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative. If x is a Vector object, this method returns a new 
        Vector with the appropriate functions performed on the Vector's 
        value and Jacobian matrix.
        '''
        try:
            return Vector(np.cosh(x._val), x._jacob*(np.sinh(x._val)))
        except AttributeError:
            try: # if scalar variable
                return Scalar(np.cosh(x._val), x._der*(np.sinh(x._val)))
   
            except AttributeError: #if constant
                return np.cosh(x)

    @staticmethod
    def tanh(x):
        '''
        Returns the tanh of a given constant or scalar or Vector object
        
        INPUTS
        =======
        x: Numeric constant or Scalar object or Vector object

        RETURNS
        =======
        tanh of numeric constant or Scalar object or Vector object
        
        NOTES
        =======
        If x is a constant, this method returns the constant sin(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative. If x is a Vector object, this method returns a new 
        Vector with the appropriate functions performed on the Vector's 
        value and Jacobian matrix.
        '''
        try:
            return Vector(np.ranh(x._val), x._jacob*(1-np.tanh(x._val)**2))
        except AttributeError:
            try: # if scalar variable
                return Scalar(np.tanh(x._val), x._der*(1-np.tanh(x._val)**2))
   
            except AttributeError: #if constant
                return np.tanh(x)


    @staticmethod
    def arcsinh(x):
        '''
        Returns the arcsinh of a given constant or scalar or Vector object
        
        INPUTS
        =======
        x: Numeric constant or Scalar object or Vector object

        RETURNS
        =======
        arcsinh of numeric constant or Scalar object or Vector object
        
        NOTES
        =======
        If x is a constant, this method returns the constant sin(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative. If x is a Vector object, this method returns a new 
        Vector with the appropriate functions performed on the Vector's 
        value and Jacobian matrix.
        '''
        try:
            return Vector(np.arcsinh(x._val), x._jacob*(-np.arcsinh(x._val)*np.arctanh(x._val)))
        except AttributeError:
            try: # if scalar variable
                return Scalar(np.arcsinh(x._val), x._der*(-np.arcsinh(x._val)*np.arctanh(x._val)))
   
            except AttributeError: #if constant
                return np.arcsinh(x)
        
        
        
    @staticmethod
    def arccosh(x):
        '''
        Returns the arccosh of a given constant or scalar or Vector object
        
        INPUTS
        =======
        x: Numeric constant or Scalar object or Vector object

        RETURNS
        =======
        arccosh of numeric constant or Scalar object or Vector object
        
        NOTES
        =======
        If x is a constant, this method returns the constant sin(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative. If x is a Vector object, this method returns a new 
        Vector with the appropriate functions performed on the Vector's 
        value and Jacobian matrix.
        '''
        try:
            j = x._jacob
            if x._val<-1 or x._val>1:
                raise ValueError('out of domain')
            else:
                return Vector(np.arccosh(x._val), x._jacob*(-np.arccosh(x._val)*np.tanh(x._val)))
        except AttributeError:
            try: # if scalar variable
                if x._val<1:
                    raise ValueError('out of domain')
                else:
                    return Scalar(np.arccosh(x._val), x._der*(-np.arccosh(x._val)*np.tanh(x._val)))
   
            except AttributeError: #if constant
                if x < 1:
                    raise ValueError('out of domain')
                else:            
                    return np.arccosh(x)
        
    @staticmethod
    def arctanh(x):
        '''
        Returns the arctanh of a given constant or scalar or Vector object
        
        INPUTS
        =======
        x: Numeric constant or Scalar object or Vector object

        RETURNS
        =======
        arctanh of numeric constant or Scalar object or Vector object 
        
        NOTES
        =======
        If x is a constant, this method returns the constant sin(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative. If x is a Vector object, this method returns a new 
        Vector with the appropriate functions performed on the Vector's 
        value and Jacobian matrix.
        '''
        try:
            j = x._jacob
            if x._val<-1 or x._val>1:
                raise ValueError('out of domain')
            else:
                return Vector(np.arctanh(x._val), x._jacob*(1-np.arctanh(x._val)**2))
        except AttributeError:
            try: 
                if x._val<-1 or x._val>1:
                    raise ValueError('out of domain')
                else:
                    return Scalar(np.arctanh(x._val), x._der*(1-np.arctanh(x._val)**2))
   
            except AttributeError: #if constant
                if x<-1 or x>1:
                    raise ValueError('out of domain')
                else:
                    return np.arctanh(x)
        

