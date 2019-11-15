from Vordiff.operator import Operator as op
from Vordiff.autodiff import AutoDiff as ad
import numpy as np


x = ad.scalar(0.5)
c = 1

def test_sin():
    
    # Scalar
    f = op.sin(x)
    assert f._val == np.sin(x._val)
    assert f._der == np.cos(x._val)*x._der

    #Constant
    assert op.sin(c) == np.sin(c)

def test_cos():
    
    # Scalar
    f = op.cos(x)
    assert f._val == np.cos(x._val)
    assert f._der == -np.sin(x._val)*x._der

    #Constant
    assert op.cos(c) == np.cos(c)
    
def test_tan():
    
    # Scalar
    f = op.tan(x)
    assert f._val == np.sin(x._val)
    assert f._der == x._der/np.cos(x._val)**2

    #Constant
    assert op.tan(c) == np.tan(c)
    
def test_arcsin():
    
    # Scalar
    f = op.arcsin(x)
    assert f._val == np.arcsin(x._val)
    assert f._der == 1/(x._der*(1-x._val**2)**.5)

    #Constant
    assert op.arcsin(c) == np.arcsin(c)
    
def test_arccos():
    
    # Scalar
    f = op.arcsin(x)
    assert f._val == np.arcsin(x._val)
    assert f._der == -x._der/(1-x._val**2)**.5

    #Constant
    assert op.arccos(c) == np.arccos(c)
    
def test_arctan():
    
    # Scalar
    f = op.arctan(x)
    assert f._val == np.arctan(x._val)
    assert f._der == x._der/(1+x._val**2)

    #Constant
    assert op.arctan(c) == np.arctan(c)
    
def test_log():
    
    # Scalar
    f = op.arctan(x)
    assert f._val == np.log(x._val)
    assert f._der == x._der/x._val

    #Constant
    assert op.log(c) == np.log(c)
    
def test_exp():
    
    # Scalar
    f = op.arctan(x)
    assert f._val == np.log(x._val)
    assert f._der == x._der*np.exp(x._val)

    #Constant
    assert op.log(c) == np.log(c)