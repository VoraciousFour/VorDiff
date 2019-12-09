#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 12:16:15 2019

@author: weiruchen
"""
import numpy as np
from VorDiff.nodes.reverse_scalar import ReverseScalar

class ReverseVector():
    
    """
    Implement the reverse mode of automatic differentiation with a vector of user
    defined variables.
    """
    
    def __init__(self, vals: list):
        
        """
        Return an ReverseVector object with a list of user specified values.
        
        INPUTS
        =======
        vals: a list object, each element is real valued numeric type
        
        RETURNS
        =======
        ReverseVector class instance
        
        NOTES
        ======
        The val class variable is initialized by user defined values.  It can
        be accessed directly but it shouldn't be modified. The children and
        gradient class variables should be private and shouldn't be
        accessed or modified by users.  Also, the children class variable is a 
        list of tuples (child, val) where child is an ReverseVector object
        and val is the derivative of the ReverseVector child with respect
        to the ReverseVector self.
        """
        
        
        self._val = np.array(vals)
        self._children = {}
        self._gradient = np.zeros(len(vals))
        self._reverse_scalars = [ReverseScalar(val) for val in vals]
        
    def __getitem__(self, idx):
        
        """
        Return the idx-th ReverseScalar object defined by the idx-th variable
        """
        
        return self._reverse_scalars[idx]
    
    def get(self):
        """
        Return the value of self ReverseVector object.
        
        INPUTS
        =======
        self: ReverseVector class instance
        
        RETURNS
        =======
        self._val: values of the list of user defined variables, user defined function,
                   or intermediate ReverseVector in the computational graph represented 
                   by the self ReverseVector object
                   
        """
        return self._val
    
    def _init_children(self):
        
        """
        initialize each child in children class variable as a list of tuples
        """
        
        self._children[self] = [(None, 1)]
        
    def compute_gradient(self, var):
        
        """
        Return the gradient of the self ReverseVector object
        
        NOTES
        ======
        the val in items of children dictionary should be a numpy array for computation
        """
        
        try:
            for child, val in self._children[var]:
                if child is not None and child._gradient.any() == 0:
                    child._gradient = child._gradient + self._gradient * val
                if child is not None:
                    child.compute_gradient(var)
        
        except KeyError:
            raise ValueError("The variable is not defined!")
            
    def __add__(self, other):
        """
        Return an ReverseVector object whose value is the sum of self and other.
        
        INPUTS
        =======
        self: ReverseVector class instance
        other: either an ReverseVector object or numeric type constant
        
        RETURNS
        =======
        child: a new ReverseVector object whose value is the sum of the value of
                  the ReverseVector self and the value of other
       
        """

        
        child = ReverseVector(self._val)
        try: # if vector
            child._val = self._val + other._val
            variables = list(self._children.keys())
            children_dict = {}
            for var in variables:
                children_dict.update({var: [(self, 1)]})
            child._children = children_dict
            
            variables = list(other._children.keys())
            for var in variables:
                if var in list(child._children.keys()):
                    child._children[var] = child._children[var] + [(other, 1)]
                else:
                    child._children[var] = [(other, 1)]
        except AttributeError: # if constant
            variables = list(self._children.keys())
            child._val = self._val + other
            children_dict = {}
            for var in variables:
                children_dict.update({var: [(self, 1)]})
            child._children = children_dict
        return child

    def __radd__(self, other):
        
        """Return an ReverseVector object with value self + other."""
        
        return self.__add__(other)

    def __sub__(self, other):
        
        """Return an ReverseVector object with value self - other."""
        
#        child = ReverseVector(self._val)
#        try: # if vector
#            child._val = self._val - other._val
#            variables = list(self._children.keys())
#            child._children = {var: [(self, 1)] for var in variables}
#            
#            variables = list(other._children.keys())
#            for var in variables:
#                if var in list(child._children.keys()):
#                    child._children[var] += [(other, -1)]
#                else:
#                    child._children[var] = [(other, -1)]
#        except AttributeError: # if constant
#            variables = list(self._children.keys())
#            child._val = self._val + other
#            child._children = {var: [(self, 1)] for var in variables}
#        return child
        
        return self + (-other)
        

    def __rsub__(self, other):
        
        """Return an ReverseVector object with value other - self."""
        
#        child = ReverseVector(self._val)
#        child._val = -self._val + other._val
#        variables = list(self._children.keys())
#        child._children = {var: [(self, -1)] for var in variables}
#        return child
        return -self + other

    def __mul__(self, other):
        """
        Return an ReverseVector object whose value is the product of self and other.
        INPUTS
        =======
        self: ReverseVector class instance
        other: either a REverseVector object or numeric type constant
        
        RETURNS
        =======
        child: a new ReverseVector object whose value is the product of the value
                  of the ReverseVector self and the value of other
        
        """
        child = ReverseVector(self._val)
        try: # if vector
            child._val = self._val * other._val
            variables = list(self._children.keys())
            children_dict = {}
            for var in variables:
                children_dict.update({var: [(self, other._val)]})
            child._children = children_dict
            
            variables = list(other._children.keys())
            for var in variables:
                if var in list(child._children.keys()):
                    child._children[var] = child._children[var] + [(other, self._val)]
                else:
                    child._children[var] = [(other, self._val)]
        except AttributeError: # if constant
            child._val = self._val * other
            variables = list(self._children.keys())
            children_dict = {}
            for var in variables:
                children_dict.update({var: [(self, other)]})
            child._children = children_dict
        return child

    def __rmul__(self, other):
        
        """Return an ReverseVector object with value self * other."""
        
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        Return an ReverseVector object whose value is the quotient of self and other.
        
        INPUTS
        =======
        self: REverseVector class instance
        other: either a ReverseVector object or numeric type constant
        
        RETURNS
        =======
        child: a new ReverseVector object whose value is the quotient of the value
                  of the ReverseVector self and the value of other
        
        """
        child = ReverseVector(self._val)
        try: # if vector
            variables = list(self._children.keys())
            child._val = self._val / other._val
            children_dict = {}
            for var in variables:
                children_dict.update({var: [(self, 1/other._val)]})
            child._children = children_dict
            
            variables = list(other._children.keys())
            for var in variables:
                if var in list(child._children.keys()):
                    child._children[var] = child._children[var] + [(other, -self._val / (other._val ** 2))]
                else:
                    child._children[var] = [(other, -self._val / (other._val ** 2))]
        except AttributeError: # if constant
            variables = list(self._children.keys())
            child._val = self._val / other
            children_dict = {}
            for var in variables:
                children_dict.update({var: [(self, 1/other)]})
            child._children = children_dict
        return child

    def __rtruediv__(self, other):
        """
        Return an ReverseVector object whose value is the quotient of other and self.
        
        INPUTS
        =======
        self: ReverseVector class instance
        other: either a ReverseVector object or numeric type constant
        
        RETURNS
        =======
        child: a new ReverseVector object whose value is the quotient of the value
                  of other nad the value of the ReverseVector self
        
        """
        variables = list(self._children.keys())
        child = ReverseVector(other / self._val)
        children_dict = {}
        for var in variables:
            children_dict.update({var: [(self, -other / (self._val ** 2))]})
        child._children = children_dict
        return child

    def __pow__(self, other):
        """
        Return an ReverseVector object with value self to the power of other.
        
        INPUTS
        =======
        self: ReverseVector class instance
        other: either a ReverseVector object or numeric type constant
        
        RETURNS
        =======
        child: a new ReverseVector object whose value is the value of the ReverseVector
                  self raised to the power of the value of other
        
        """
        child = ReverseVector(self._val)
        try: # if vector
            variables = list(self._children.keys())
            child._val = self._val ** other._val
            children_dict = {}
            for var in variables:
                children_dict.update({var: [(self, other._val*self._val**(other._val-1))]})
            child._children = children_dict
            
            variables = list(other._children.keys())
            for var in variables:
                if var in list(child._children.keys()):
                    child._children[var] = child._children[var] + [(other, self._val ** other._val * np.log(self._val))]
                else:
                    child._children[var] = [(other, self._val ** other._val * np.log(self._val))]
        except AttributeError: # if constant
            variables = list(self._children.keys())
            child._val = self._val ** other
            children_dict = {}
            for var in variables:
                children_dict.update({var: [(self, other*self._val ** (other - 1))]})
            child._children = children_dict
        return child

    def __rpow__(self, other):
        """
        Return an ReverseVector object with value other to the power of self.
        
        INPUTS
        =======
        self: ReverseVector class instance
        other: either an ReverseVector object or numeric type constant
        
        RETURNS
        =======
        child: a new ReverseVector object whose value is the value of other raised
                  to the power of the value of the ReverseVector self
        
        """
        variables = list(self._children.keys())
        child = ReverseVector(other ** self._val)
        children_dict = {}
        for var in variables:
            children_dict.update({var: [(self, other ** self._val * np.log(other))]})
        child._children = children_dict
        return child

    def __neg__(self):
        """
        Return an ReverseVector object with the negated value of self.
        
        INPUTS
        =======
        self: ReverseVector class instance
        
        RETURNS
        =======
        A new ReverseVector object whose value is the negated value of the ReverseVector self
        
        """
        variables = list(self._children.keys())
        child = ReverseVector(-self._val)
        children_dict = {}
        for var in variables:
            children_dict.update({var: [(self, -1)]})
        child._children = children_dict
        return child
    
    


