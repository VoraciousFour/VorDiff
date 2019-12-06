from VorDiff.nodes.scalar import Scalar
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

    @staticmethod
    def vector(vec):
        '''
        Creates a Vector object with the vector given and jthe acobian matrix
        
        INPUTS
        =======
        val: The numeric values at which to evaluate
        
        RETURNS
        =======
        Vector objects
        '''
        return Vector(vec, np.eye(len(vec)))
