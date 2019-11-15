#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:09:28 2019

@author: weiruchen
"""

import numpy as np

class Scalar():
    
    """
    Implements the interface for user defined variables, which is single value in this case.
    So the scalar objects have a single user defined value (or it can be a single 
    function composition) and a derivative. For the single valued scalar case, the jacobian 
    matrix is just simply the derivative.
    """

    def __init__(self, value, *kwargs):
        """
        Return a Scalar object with user defined value and derivative.
        If no user defined derivative is defined, then just return a Scalar object with user defined value.
        
        INPUTS
        =======
        val: real valued numeric type
        der: real valued numeric type
        
        RETURNS
        =======
        Scalar object
        
        NOTES
        ======
        User can access and modify _val and _der class variables directly. 
        To access these two variables, user can call the get method in the Scalar class.
        """
        
        self._val = value
        if len(kwargs) == 0:
            self._der = 1
        else:
            self._der = kwargs[0]
    
    def get(self):
        """
        Return the value and derivative of the self Scalar object.
        
        INPUTS
        =======
        self: Scalar object
        
        RETURNS
        =======
        self._val: value of the user defined variable or value of the function 
                   represented by the self Scalar object
        self._der: the derivative of the self Scalar object with respect to the single variable
        """
        return self._val, self._der

    def __add__(self, other):
        """
        Return a Scalar object whose value is the sum of self and other 
        when other is a Scalar object.
        
        INPUTS
        =======
        self: Scalar object
        other: Scalar object
        
        RETURNS
        =======
        a new Scalar object whose value is the sum of the values of
        the Scalar self and other and whose derivative is the new derivative of the function
        that sums up these two values with respect to the single variable.
        """
        try:
            return Scalar(self._val+other._val, self._der+other._der)
        except AttributeError:
            return self.__radd__(other)

    def __radd__(self, other):
        """
        Return a Scalar object whose value is the sum of self and other 
        when other is a numeric type constant.
        """
        return Scalar(self._val+other, self._der)

    
    def __mul__(self, other):
        """
        Return a Scalar object whose value is the product of self and other
        when other is a Scalar object.
        
        INPUTS
        =======
        self: Scalar object
        other: Scalar object
        
        RETURNS
        =======
        a new Scalar object whose value is the product of the values of 
        the Scalar self and other and whose derivative is the new derivative of the function
        that multiplies these two values with respect to the single variable.
        
        """
        try:
            return Scalar(self._val*other._val, self._der*other._val+self._val*other._der)
        except AttributeError:
            return self.__rmul__(other)
    
    def __rmul__(self, other):
        """
        Return a Scalar object whose value is the product of self and other
        when other is a numeric type constant.
        """
        return Scalar(self._val*other, self._der*other)
    
    def __sub__(self, other):
        """Return a Scalar object with value self - other"""
        return self + (-other)
        
    
    def __rsub__(self, other):
        """Return a Scalar object with value other - self"""
        return -self + other
    
    def __truediv__(self, other):
        """
        Scalar value is returned from one user defined variable
        created by the quotient of self and other. 
        
        INPUTS
        =======
        self: Scalar object
        other: either a Scalar object or numeric type constant
        
        RETURNS
        =======
        a new Scalar object whose value is the quotient of the values of
        the Scalar self and other and whose derivative is the new derivative of the function
        that divides Scalar self by other with respect to the single variable.
        """
        try:
            return Scalar(self._val/other._val, (self._der*other._val-self._val*other._der)/(other._val**2))
        except AttributeError:
            return Scalar(self._val/other, self._der/other)
    
    def __rtruediv__(self, other):
        """
        Return a Scalar object whose value is the quotient of self and other
        when other is a numeric type constant.
        """
        return Scalar(other/self._val, other*(-self._der)/(self._val)**2)

        
    def __pow__(self, other):
        
        """
        INPUTS
        =======
        self: Scalar object
        other: either a Scalar object or numeric type constant
        
        RETURNS
        =======
        Scalar object whose value is self raised to the power of other
        
        NOTES
        ======
        This method returns a Scalar object that is calculated from the 
        self Scalar class instance raised to the power of other
        """
        
        try:
            return Scalar(self._val**other._val, (other._val*self._der/self._val+np.log(self._val)*other._der)*(self._val**other._val))
        except AttributeError:
            return Scalar(self._val**other, other*(self._val**(other-1))*self._der)
            
    def __rpow__(self, other):
        """
        Return a Scalar object that is calculated by taking the value of other
        and raising it to the power of self when other is a numeric type constant.
        """
        return Scalar(other**self._val, (other**self._val)*np.log(other)*self._der)
    
    def __neg__(self):
        """
        INPUTS
        =======
        self: Scalar object
        
        RETURNS
        =======
        A Scalar object that is the negation of self. 
        
        NOTES
        ======
        The Scalar Object that is returned from this method comes from 
        a new Scalar Object that is the negation of self. 
        """

        return Scalar((-1)*self._val, (-1)*self._der)
    
