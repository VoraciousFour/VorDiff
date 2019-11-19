class Scalar():
    
    def __init__(self, val, *kwargs):

        # Initialize Scalar Object with Value given by the User
        if len(kwargs) == 0:
            self._der = 1
        self._val = val
    

    def __add__(self, other):
        # Returns Scalar that is the sum of self and other value 
        try:
            return Scalar(self._val + other._val, self._der + other._der)
        except AttributeError:
            # Or use reverse addition 
            return self.__radd__(other)

    def __radd__(self, other):
        return Scalar(self._val + other , self._der)

    def __mul__(self, other):
         # Returns Scalar that is the product of self and other value 
        try:
            return Scalar(self._val * other._val, self._der * other._val + self._val * other._der)
        except AttributeError:
            return self.__rmul__(other)

    def __rmul__(self, other): 
        return Scalar(self._val*other, self._der*other)

    def __sub__(self, other):
        """ Returns Scalar whose value is self subtracted by other """ 
        try:
            return Scalar(self._val - other, self._der - other)
        except AttributeError: 
            return self.__rsub__(other)

    def __rsub__(self, other):
        """ Returns Scalar whose value is other subtracted by self """
        return Scalar(self._val - other, self._der - other)

    def __truediv__(self, other):
        """
        INPUTS
        =======
        self: Scalar class instance
        other: Scalar object or numeric constant

        RETURNS
        =======
        Scalar whose value is the quotient of self and other

        NOTES
        ======
        Scalar value is returned from one user defined variable
        created by the quotient of self and other. 
        """
        try:
            return Scalar(self._val / other._val, self._der / other._val)
        except AttributeError:
            return self.__rtruediv__(other)

    def __rtruediv__(self, other): 
        """
        INPUTS
        =======
        self: Scalar class instance
        other: Scalar object or numeric constant

        RETURNS
        =======
        Returns Scalar whose value is the quotient of other and self

        NOTES
        ======
        Scalar value is returned from one user defined variable
        created by the quotient of other and self. 

        """
        return Scalar(other / self._val, other / self._der)

    def __neg__(self): 
        """
        INPUTS
        =======
        self: Scalar class instance

        RETURNS
        =======
        A Scalar object that is the negation of self. 

        NOTES
        ======
        The Scalar Object that is returned from this method comes from a new Scalar Object that is the negation of self. 
        """
        return Scalar(-1 * self._val, -1* self._der)

    def __pow__(self, other):
        """
        INPUTS
        =======
        self: Scalar class instance
        other: Scalar object or numeric constant

        RETURNS
        =======
        Scalar object whose value is self raised to the power of other

        NOTES
        ======
        This method returns a Scalar object that is calculated from the self Scalar class instance raised to the power of other
        """
        # Returns Scalar whose value is self raised to power of other 
        try: 
            return Scalar(self._val ** other._val, self._der ** other._der)
        except AttributeError: 
            return self.__rpow__(other)
                
    def __rpow__(self, other): 
        """
        INPUTS
        =======
        self: Scalar class instance
        other: Scalar object or numeric constant

        RETURNS
        =======
        A Scalar object whose value is the value of other raised
        to the power of self, the Scalar class instance. 

        NOTES
        ======
        The method returns a Scalar object that is calculated by taking the value of other
        and raising it to the power of self. 
        """
        # Returns Scalar whose value is other to power of self
        return Scalar(other ** self._val, other ** self._der)

# a = 2.0 # Value to evaluate at


# alpha = 2.0
# beta = 3.0

# x = AutoDiffToy(a)
# f = alpha * x + beta

# print(f.value, f.derivative)

# f1 = alpha * x + beta
# f2 = x * alpha + beta
# f3 = beta + alpha * x
# f4 = beta + x * alpha

# print(f1.value, f1.derivative)
# print(f2.value, f2.derivative)
# print(f3.value, f3.derivative)
# print(f4.value, f4.derivative)
