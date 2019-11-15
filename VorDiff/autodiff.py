from nodes.scalar import Scalar


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
        
        return Scalar(val=val, der=1)
