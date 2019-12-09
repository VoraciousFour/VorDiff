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
    
    @staticmethod
    def reverse_vector(vals):
        '''
        Creates a ReverseVector object with the values given
        
        INPUTS
        =======
        vals: The list of numeric values at which to evaluate
        
        RETURNS
        =======
        ReverseVector objects
        '''
        reverse_vecs = [None] * len(vals)
        for i in range(len(vals)):
            reverse_vecs[i] = ReverseVector(vals[i])
            reverse_vecs[i]._init_children()
        return reverse_vecs

    def partial(f, x):
        
        f._gradient = 1
        f.compute_gradient(x)
        return x._gradient
