from VorDiff.nodes.scalar import Scalar
from VorDiff.nodes.vector import Element
from VorDiff.nodes.vector import Vector
import numpy as np


class AutoDiff():
    '''
    The AutoDiff class allows users to define Scalar variables and 
    interface with the auto-differentiator.
    '''
    
    @staticmethod
    def scalar(val):
        '''
        Creates a Scalar object with the value given and derivative 1
        
        INPUTS
        =======
        val: The numeric value at which to evaluate
        
        RETURNS
        =======
        Scalar objects
        '''
        
        return Scalar(val, 1)

    def element(val,jacob):
        '''
        Creates an Element object with the value given and jacobian matrix 

        INPUTS
        =======
        val: The numeric value of the function
        jacob: The jacobian matrix value of the function at which to evaluate

        RETURNS
        =======
        Element objects
        '''
        return Element(val,jacob)


    @staticmethod
    def vector(vec):
        '''
        Creates a Vector object with the vector given and the jacobian matrix
        
        INPUTS
        =======
        val: The numeric values at which to evaluate
        
        RETURNS
        =======
        Vector objects
        '''
        return Vector(vec, np.eye(len(vec)))

