#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 13:09:28 2019

@author: weiruchen
"""



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
        try:
            return Scalar(self._val-other._val, self._der-other._der)
        except AttributeError:
            return self.__rsub__(other)
    
    def __rsub__(self, other):
        return Scalar(self._val-other, self._der)
    
    def __truediv__(self, other):
        try:
            return Scalar(self._val/other._val, -(self._val*other._der-self._der*other._val)/(other._val**2))
        except AttributeError:
            return self.__rtruediv__(other)
    
    def __rtruediv__(self, other):
        return Scalar(self._val/other, self._der/other)
        
    def __pow__(self, other):
        try:
            return Scalar(self._val**other._val, other._val*(self._val)**(other._val-1)*self._der)
        except AttributeError:
            return self.__rpow__(other)
    
    def __rpow__(self, other):
        return Scalar(self._val**other, other*(self._val)**(other-1)*self._der)
    
    def __neg__(self):
        return Scalar((-1)*self._val, (-1)*self._der)
    
