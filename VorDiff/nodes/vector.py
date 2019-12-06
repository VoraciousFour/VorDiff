import numpy as np

class Vector():
    
    """
    Implements the interface for user defined variables, which is list of values in this case.
    So the Vector objects have a list of user defined values
    and a list of derivatives with respect to each variable.
    """

    def __init__(self, vec, *kwargs):
        """
        Return a Vector object with user defined values and derivatives.
        If no user defined derivatives are defined, then just return a Vector object with 
        user defined values and an identity matrix with size len(values) by len(values)
        as the jacobian matrix.
        
        INPUTS
        =======
        vec: list object, each element is real valued numeric type
        jacob: 2d list object, each element is real valued numeric type
        
        RETURNS
        =======
        Vector object
        
        NOTES
        ======
        User can access and modify _vec and _jacob class variables directly. 
        To access these two variables, user can call the get method in the Vector class.
        """
        
        self._vec = np.array(vec)
        if len(kwargs) == 0:
            self._jacob = np.eye(len(vec))
        else:
            self._jacob = np.array(kwargs[0])
            
        elements = []
        for i in range(len(vec)):
            elements.append(Element(self._vec[i], self._jacob[i]))
        self._elements = elements
        
    def __getitem__(self, idx):
        """
        Return the idx-th Element object defined by the idx-th variable
        """
        return self._elements[idx]
    
#    def get(self):
#        """
#        Return the vector of values and derivatives (jacobian matrix) of the self Vector object.
#        
#        INPUTS
#        =======
#        self: Vector object
#        
#        RETURNS
#        =======
#        self._vec: vector of values of the user defined variables
#        self._jacob: the derivatives of the self Vector object with respect to the list of
#        variables (the jacobian matrix)
#        """
#        return self._vec, self._jacob

class Element():
    
    """
    The Element object has an evaluated value (it can be the value of function compositions with
    the input of user defined values) and a list of current derivatives with respect to each variable.
    """
    
    def __init__(self, val, jacob):
        """
        Return an Element object with an evaluated value and a list of current derivatives
        with respect to each user defined variable.
        
        
        INPUTS
        =======
        val: real valued numeric type
        jacob: list object, each element is real valued numeric type
        
        RETURNS
        =======
        Element object
        
        NOTES
        ======
        User can access and modify _val and _jacob class variables directly. 
        To access these two variables, user can call the get method in the Element class.
        """
        self._val = val
        self._jacob = np.array(jacob)
        
#    def get(self):
#        """
#        Return the value and the list of derivatives of the self Element object.
#        
#        INPUTS
#        =======
#        self: Element object
#        
#        RETURNS
#        =======
#        self._val: value of the function represented by the self Element object
#        self._jacob: the list of derivatives of the self Element object with respect to 
#        each user defined variable
#        """
#        return self._val, self._jacob
    
    def get_val(self):
        """
        Return the value of the self Element object.
        
        INPUTS
        =======
        self: Element object
        
        RETURNS
        =======
        self._val: value of the function represented by the self Element object
        """
        return self._val
    
    def get_derivatives(self):
        """
        Return the list of derivatives of the self Element object.
        
        INPUTS
        =======
        self: Element object
        
        RETURNS
        =======
        self._jacob: the list of derivatives of the self Element object with respect to 
        each user defined variable
        """
        return self._jacob
        

    def __add__(self, other):
        """
        Return an Element object whose value is the sum of self and other 
        when other is an Element object.
        
        INPUTS
        =======
        self: Element object
        other: Element object
        
        RETURNS
        =======
        a new Element object whose value is the sum of the values of
        the Element self and other and whose derivatives are the new list of 
        derivatives of the function that sums up these two values 
        with respect to each user defined variable.
        """
        try:
            val = self._val+other._val
            jacob = self._jacob+other._jacob
            return Element(val, jacob)
        
        except AttributeError:
            return self.__radd__(other)

    def __radd__(self, other):
        """
        Return an Element object whose value is the sum of self and other 
        when other is a numeric type constant.
        """
        return Element(self._val+other, self._jacob)

    
    def __mul__(self, other):
        """
        Return an Element object whose value is the product of self and other
        when other is an Element object.
        
        INPUTS
        =======
        self: Element object
        other: Element object
        
        RETURNS
        =======
        a new Element object whose value is the product of the values of 
        the Element self and other and whose derivatives are the new list of 
        derivatives of the function that multiplies these two values 
        with respect to each user defined variable.
        
        """
        try:
            val = self._val*other._val
            jacob = other._val*self._jacob+self._val*other._jacob
            return Element(val, jacob)
        
        except AttributeError:
            return self.__rmul__(other)
    
    def __rmul__(self, other):
        """
        Return an Element object whose value is the product of self and other
        when other is a numeric type constant.
        """
        return Element(self._val*other, self._jacob*other)
    
    def __sub__(self, other):
        """Return an Element object with value self - other"""
        return self + (-other)
        
    
    def __rsub__(self, other):
        """Return an Element object with value other - self"""
        return -self + other
    
    def __truediv__(self, other):
        """
        Return an Element object whose value is the quotient of self and other. 
        
        INPUTS
        =======
        self: Element object
        other: either an Element object or numeric type constant
        
        RETURNS
        =======
        a new Element object whose value is the quotient of the values of
        the Element self and other and whose derivatives are the list of 
        new derivatives of the function that divides Element self by other 
        with respect to each user defined variable.
        """
        try:
            val = self._val/other._val
            jacob = (self._jacob*other._val-self._val*other._jacob)/(other._val**2)
            return Element(val, jacob)
        
        except AttributeError:
            return Element(self._val/other, self._jacob/other)
    
    def __rtruediv__(self, other):
        """
        Return an Element object whose value is the quotient of other and self
        when other is a numeric type constant.
        """
        return Element(other/self._val, other*(-self._jacob)/(self._val)**2)

        
    def __pow__(self, other):
        
        """
        INPUTS
        =======
        self: Element object
        other: either an Element object or numeric type constant
        
        RETURNS
        =======
        Element object whose value is self raised to the power of other
        
        NOTES
        ======
        This method returns an Element object that is calculated from the 
        self Element class instance raised to the power of other
        """
        try:
            val = self._val**other._val
            jacob = np.exp(other._val*np.log(self._val))*(other._jacob*np.log(self._val)+other._val/float(self._val)*self._jacob)
            return Element(val, jacob)
        
        except AttributeError:
            return Element(self._val**other, other*(self._val**(other-1))*self._jacob)
        
    
    def __rpow__(self, other):
        """Return an Element object that is calculated from other raised to the power of self"""
        return Element(other**self._val, self._jacob*np.log(other)*other**self._val)
         
    
    def __neg__(self):
        """
        INPUTS
        =======
        self: Element object
        
        RETURNS
        =======
        An Element object that is the negation of self. 
        
        NOTES
        ======
        The Element object that is returned from this method comes from 
        a new Element object that is the negation of self. 
        """

        return Element((-1)*self._val, (-1)*self._jacob)
    
    def __eq__(self, other):
        """
        INPUTS
        =======
        self: Element object
        other: Element object or numeric type constant
        
        RETURNS
        =======
        A boolean that indicates if the self object and other object are equal. 
        """
        try:
            return self._val == other._val and (self._jacob == other._jacob).all()
        except AttributeError:
            return False
    
    def __ne__(self, other):
        """
        INPUTS
        =======
        self: Element object
        other: Element object or numeric type constant
        
        RETURNS
        =======
        A boolean that indicates if the self object and other object are not equal. 
        """
        try:
            return self._val != other._val or (self._jacob != other._jacob).any()
        except AttributeError:
            return True
    



