from VorDiff.nodes.reverse_scalar import ReverseScalar

class ReverseAutoDiff():
    '''
    The AutoDiff class allows users to define Scalar variables and 
    interface with the auto-differentiator.
    '''
    
    @staticmethod
    def reverse_scalar(val):
        '''
        Creates a Scalar object with the value given and derivative 1
        
        INPUTS
        =======
        val: The numeric value at which to evaluate
        
        RETURNS
        =======
        Scalar objects
        '''
        
        return ReverseScalar(val)