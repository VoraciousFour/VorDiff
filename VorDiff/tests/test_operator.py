from Vordiff.operator import Operator as op
from Vordiff.autodiff import AutoDiff as ad
import numpy as np


x = ad.scalar(0.5)
c = 1

def test_sin():
    # Scalar
    f = op.sin(x)
    assert f._val == np.sin(x._val)
    assert f._der == np.cos(x._val)*x_der

    #Constant
    assert op.sin(c) == np.sin(c)

