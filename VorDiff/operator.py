import numpy as np
import math
from VorDiff.nodes.scalar import Scalar
from VorDiff.nodes.vector import Vector,Element


class Operator():
    
    @staticmethod
    def sin(x):
        '''
        Returns the sine of a given constant, Scalar, or Element object
        
        INPUTS
        =======
        x: Numeric constant, Scalar object or Element object

        RETURNS
        =======
        sine of numeric constant, Scalar object or Element object
        
        NOTES
        =======
        If x is a constant, this method returns the constant sin(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative. If x is a Element object, this method returns a new 
        Element with the appropriate functions performed on the Element's 
        value and Jacobian matrix.
        '''
        
        try:
            return Element(np.sin(x._val), np.cos(x._val)*x._jacob)
        except AttributeError: # If contant
            try: # If scalar variable
                return Scalar(np.sin(x._val), x._der*np.cos(x._val))
            
            except AttributeError: # If contant
                return np.sin(x)
        
    @staticmethod
    def cos(x):
        '''
        Returns the cosine of a given constant, scalar or Element
        
        INPUTS
        =======
        x: Numeric constant, Scalar object or Element object

        RETURNS
        =======
        cosine of numeric constant or Scalar object or Element object
        
        NOTES
        =======
        If x is a constant, this method returns the constant sin(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative. If x is a Element object, this method returns a new 
        Element with the appropriate functions performed on the Element's 
        value and Jacobian matrix.
        '''
        
        try:
            return Element(np.cos(x._val), -np.sin(x._val)*x._jacob)
        except AttributeError: # If contant
            try: # If scalar variable
                return Scalar(np.cos(x._val), -np.sin(x._val)*x._der)
            
            except AttributeError: # If contant
                return np.cos(x)
        
    @staticmethod
    def tan(x):
        '''
        Returns the tangent of a given constant or scalar or Element object
        
        INPUTS
        =======
        x: Numeric constant or Scalar object or Element object

        RETURNS
        =======
        tangent of numeric constant or Scalar object or Element object
        
        NOTES
        =======
        If x is a constant, this method returns the constant sin(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative. If x is a Element object, this method returns a new 
        Element with the appropriate functions performed on the Element's 
        value and Jacobian matrix.
        ''' 
        try:
            return Element(np.tan(x._val), x._jacob/np.cos(x._val)**2)
        except AttributeError: # If contant
            try: # If scalar variable
                return Scalar(np.tan(x._val), x._der/np.cos(x._val)**2)
            
            except AttributeError: # If contant
                return np.tan(x)
        
    @staticmethod
    def arcsin(x):
        '''
        Returns the arcsine of a given constant or scalar or Element object
        
        INPUTS
        =======
        x: Numeric constant or Scalar object or Element object
 
        RETURNS
        =======
        arcsine of numeric constant or Scalar object or Element object
        
        NOTES
        =======
        If x is a constant, this method returns the constant sin(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative. If x is a Element object, this method returns a new 
        Element with the appropriate functions performed on the Element's 
        value and Jacobian matrix.
        '''
        
        try:
            j = x._jacob
            if x._val<-1 or x._val>1:
                raise ValueError('out of domain')
            else:
                return Element(np.arcsin(x._val), 1/(x._jacob*(1-x._val**2)**.5))
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
        Returns the arccosine of a given constant or scalar or Element object
        
        INPUTS
        =======
        x: Numeric constant or Scalar object or Element object

        RETURNS
        =======
        arccosine of numeric constant or Scalar object or Element object
        
        NOTES
        =======
        If x is a constant, this method returns the constant sin(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative. If x is a Element object, this method returns a new 
        Element with the appropriate functions performed on the Element's 
        value and Jacobian matrix.
        '''
        
        try:
            j = x._jacob
            if x._val<-1 or x._val>1:
                raise ValueError('out of domain')
            else:
                return Element(np.arccos(x._val), -x._jacob/(1-x._val**2)**.5)
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
        Returns the arctangent of a given constant or scalar or Element object
        
        INPUTS
        =======
        x: Numeric constant or Scalar object or Element object

        RETURNS
        =======
        arctangent of numeric constant or Scalar object or Element object
        
        NOTES
        =======
        If x is a constant, this method returns the constant sin(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative. If x is a Element object, this method returns a new 
        Element with the appropriate functions performed on the Element's 
        value and Jacobian matrix.
        '''
        
        try:
            return Element(np.arctan(x._val), x._jacob/(1+x._val**2))
        except AttributeError:
            try: # If scalar variable
                return Scalar(np.arctan(x._val), x._der/(1+x._val**2))
            
            except: # If contant
                return np.arctan(x)
        
    @staticmethod
    def log(*args):
        '''
        Returns the log of a given constant or scalar or Element object
        
        INPUTS
        =======
        if the lenth of input is 2:
        a: Numeric constant
        x: Numeric constant or Scalar object or Element object

        if the lenth of input is 1:
        x: Numeric constant or Scalar object or Element object

        RETURNS
        =======
        log of numeric constant or Scalar object or Element object
        
        NOTES
        =======
        If x is a constant, this method returns the constant sin(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative. If x is a Element object, this method returns a new 
        Element with the appropriate functions performed on the Element's 
        value and Jacobian matrix.
        '''
        if len(args) == 2:
        	return log_(args[0],args[1])
        else:
            x = args[0]
            try:
                j = x._jacob
                if x._val<=0:
                    raise ValueError('out of domain')
                else:
                    return Element(np.log(x._val), x._jacob/x._val)
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
    def exp(*args):
        '''
        Returns the exponential of a given constant or scalar or Element object
        
        INPUTS
        =======
        if the lenth of input is 2:
        a: Numeric constant
        x: Numeric constant or Scalar object or Element object
        
        if the lenth of input is 1:
        x: Numeric constant or Scalar object or Element object

        RETURNS
        =======
        exponential of numeric constant or Scalar object or Element object
        
        NOTES
        =======
        If x is a constant, this method returns the constant sin(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative. If x is a Element object, this method returns a new 
        Element with the appropriate functions performed on the Element's 
        value and Jacobian matrix.
        '''
        if len(args) == 2:
        	return exp_(args[0],args[1])
        else:
            x = args[1]
            try:
                 return Element(np.exp(x._val), x._jacob*(np.exp(x._val)))
            except AttributeError:
                try: # If scalar variable
                    return Scalar(np.exp(x._val), x._der*np.exp(x._val))
            
                except AttributeError: # If contant
                    return np.exp(x)
        
    @staticmethod
    def sinh(x):
        '''
        Returns the sinh of a given constant or scalar or Element object
        
        INPUTS
        =======
        x: Numeric constant or Scalar object or Element object

        RETURNS
        =======
        sinh of numeric constant or Scalar object or Element object
        
        NOTES
        =======
        If x is a constant, this method returns the constant sin(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative. If x is a Element object, this method returns a new 
        Element with the appropriate functions performed on the Element's 
        value and Jacobian matrix.
        '''
        try:
            return Element(np.sinh(x._val), x._jacob*(np.cosh(x._val)))
        except AttributeError:
        
            try: # if scalar variable
                return Scalar(np.sinh(x._val), x._der*(np.cosh(x._val)))
   
            except AttributeError: #if constant
                return np.sinh(x)  
      
    @staticmethod
    def cosh(x):
        '''
        Returns the cosh of a given constant or scalar or Element object
        
        INPUTS
        =======
        x: Numeric constant or Scalar object or Element object

        RETURNS
        =======
        cosh of numeric constant or Scalar object or Element object
        
        NOTES
        =======
        If x is a constant, this method returns the constant sin(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative. If x is a Element object, this method returns a new 
        Element with the appropriate functions performed on the Element's 
        value and Jacobian matrix.
        '''
        try:
            return Element(np.cosh(x._val), x._jacob*(np.sinh(x._val)))
        except AttributeError:
            try: # if scalar variable
                return Scalar(np.cosh(x._val), x._der*(np.sinh(x._val)))
   
            except AttributeError: #if constant
                return np.cosh(x)

    @staticmethod
    def tanh(x):
        '''
        Returns the tanh of a given constant or scalar or Element object
        
        INPUTS
        =======
        x: Numeric constant or Scalar object or Element object

        RETURNS
        =======
        tanh of numeric constant or Scalar object or Element object
        
        NOTES
        =======
        If x is a constant, this method returns the constant sin(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative. If x is a Element object, this method returns a new 
        Element with the appropriate functions performed on the Element's 
        value and Jacobian matrix.
        '''
        try:
            return 	Element(np.ranh(x._val), x._jacob*(1-np.tanh(x._val)**2))
        except AttributeError:
            try: # if scalar variable
                return Scalar(np.tanh(x._val), x._der*(1-np.tanh(x._val)**2))
   
            except AttributeError: #if constant
                return np.tanh(x)


    @staticmethod
    def arcsinh(x):
        '''
        Returns the arcsinh of a given constant or scalar or Element object
        
        INPUTS
        =======
        x: Numeric constant or Scalar object or Element object

        RETURNS
        =======
        arcsinh of numeric constant or Scalar object or Element object
        
        NOTES
        =======
        If x is a constant, this method returns the constant sin(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative. If x is a Element object, this method returns a new 
        Element with the appropriate functions performed on the Element's 
        value and Jacobian matrix.
        '''
        try:
            return Element(np.arcsinh(x._val), x._jacob*(-np.arcsinh(x._val)*np.arctanh(x._val)))
        except AttributeError:
            try: # if scalar variable
                return Scalar(np.arcsinh(x._val), x._der*(-np.arcsinh(x._val)*np.arctanh(x._val)))
   
            except AttributeError: #if constant
                return np.arcsinh(x)
        
        
        
    @staticmethod
    def arccosh(x):
        '''
        Returns the arccosh of a given constant or scalar or Element object
        
        INPUTS
        =======
        x: Numeric constant or Scalar object or Element object

        RETURNS
        =======
        arccosh of numeric constant or Scalar object or Element object
        
        NOTES
        =======
        If x is a constant, this method returns the constant sin(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative. If x is a Element object, this method returns a new 
        Element with the appropriate functions performed on the Element's 
        value and Jacobian matrix.
        '''
        try:
            j = x._jacob
            if x._val<-1 or x._val>1:
                raise ValueError('out of domain')
            else:
                return 	Element(np.arccosh(x._val), x._jacob*(-np.arccosh(x._val)*np.tanh(x._val)))
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
        Returns the arctanh of a given constant or scalar or Element object
        
        INPUTS
        =======
        x: Numeric constant or Scalar object or Element object

        RETURNS
        =======
        arctanh of numeric constant or Scalar object or Element object 
        
        NOTES
        =======
        If x is a constant, this method returns the constant sin(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative. If x is a Element object, this method returns a new 
        Element with the appropriate functions performed on the Element's 
        value and Jacobian matrix.
        '''
        try:
            j = x._jacob
            if x._val<-1 or x._val>1:
                raise ValueError('out of domain')
            else:
                return Element(np.arctanh(x._val), x._jacob*(1-np.arctanh(x._val)**2))
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

    @staticmethod
    def logistic(x):
        '''
        Returns the logistic of a given constant or scalar or Element object
        
        INPUTS
        =======
        x: Numeric constant or Scalar object or Element object

        RETURNS
        =======
        logistic of numeric constant or Scalar object or Element object 
        
        NOTES
        =======
        If x is a constant, this method returns the constant logistic(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative. If x is a Element object, this method returns a new 
        Element with the appropriate functions performed on the Element's 
        value and Jacobian matrix.
        '''
        try:
            return Element(1/(1+np.exp(-x._val)), x._jacob*(x._val**2*np.exp(-x._val)))
        except AttributeError:
            try: 
                return Scalar(1/(1+np.exp(-x._val)), x._der*(x._val**2*np.exp(-x._val)))
   
            except AttributeError: #if constant
                return 1/(1+np.exp(-x._val))



    @staticmethod
    def square_root(x):
        '''
        Returns the square root of a given constant or scalar or Element object
        
        INPUTS
        =======
        x: Numeric constant or Scalar object or Element object

        RETURNS
        =======
        square root of numeric constant or Scalar object or Element object 
        
        NOTES
        =======
        If x is a constant, this method returns the constant square root(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative. If x is a Element object, this method returns a new 
        Element with the appropriate functions performed on the Element's 
        value and Jacobian matrix.
        '''
        try:
            j = x._jacob
            if x._val<=0:
                raise ValueError('out of domain')
            else:
                return Element(x._val**0.5, x._jacob*(x._val**(-1/2)/2))
        except AttributeError:
            try: 
                if x._val<=0:
                    raise ValueError('out of domain')
                else:
                    return Scalar(x._val**0.5, x._der*(x._val**(-1/2)/2))
   
            except AttributeError: #if constant
                if x<=0:
                    raise ValueError('out of domain')
                else:
                    return x**0.5

    @staticmethod
    def log_(a,x):
        '''
        Returns the log of a given constant or scalar or Element object with base a
        
        INPUTS
        =======
        x: Numeric constant or Scalar object or Element object
        a: Numeric constant

        RETURNS
        =======
        log of numeric constant or Scalar object or Element object
        
        NOTES
        =======
        If x is a constant, this method returns the constant sin(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative. If x is a Element object, this method returns a new 
        Element with the appropriate functions performed on the Element's 
        value and Jacobian matrix.
        '''
        
        try:
            j = x._jacob
            if x._val<=0:
                raise ValueError('out of domain')
            else:
                return Element(math.log(a, x._val), x._jacob/(x._val*np.log(a)))
        except AttributeError:
            try: # If scalar variable
                if x._val<=0:
                    raise ValueError('out of domain')
                else:    
                    return Scalar(math.log(a, x._val), x._der/(x._val*np.log(a)))
            
            except AttributeError: # If contant
                if x<=0:
                    raise ValueError('out of domain')
                else:
                    return math.log(a,x)
        
    @staticmethod
    def exp_(a, x):
        '''
        Returns the exponential of a given constant or scalar or Element object with base a
        
        INPUTS
        =======
        x: Numeric constant or Scalar object or Element object
        a: Numeric constant

        RETURNS
        =======
        exponential of numeric constant or Scalar object or Element object
        
        NOTES
        =======
        If x is a constant, this method returns the constant sin(x). If
        x is a Scalar object, this method returns a new Scalar with the
        appropriate functions performed on the Scalar's value and
        derivative. If x is a Element object, this method returns a new 
        Element with the appropriate functions performed on the Element's 
        value and Jacobian matrix.
        '''
        
        try:
            return Element(a**x._val, x._jacob*np.log(a)*(a**x._val))
        except AttributeError:
            try: # If scalar variable
                return Scalar(a**x._val, x._der*np.log(a)*(a**x._val))
            
            except AttributeError: # If contant
                return a**x
        
