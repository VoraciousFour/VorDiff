import sys
sys.path.append('..')
from VorDiff.operator import Operator as op
from VorDiff.autodiff import AutoDiff as ad
import numpy as np


x = ad.scalar(0.5)
c = 0.5
y = ad.scalar(2.0)
d = 2.0

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
    assert f._val == np.tan(x._val)
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
    f = op.arccos(x)
    assert f._val == np.arccos(x._val)
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
    f = op.log(x)
    assert f._val == np.log(x._val)
    assert f._der == x._der/x._val

    #Constant
    assert op.log(c) == np.log(c)
    
def test_exp():
    
    # Scalar
    f = op.exp(x)
    assert f._val == np.exp(x._val)
    assert f._der == x._der*np.exp(x._val)

    #Constant
    assert op.exp(c) == np.exp(c)


def test_sinh():
    # Scalar
    f = op.sinh(x)
    assert f._val == np.sinh(x._val)
    assert f._der == np.cosh(x._val)*x._der

    #Constant
    assert op.sinh(c) == np.sinh(c)

def test_cosh():
    # Scalar
    f = op.cosh(x)
    assert f._val == np.cosh(x._val)
    assert f._der == np.sinh(x._val)*x._der

    #Constant
    assert op.cosh(c) == np.cosh(c)

def test_tanh():
    # Scalar
    f = op.tanh(x)
    assert f._val == np.tanh(x._val)
    assert f._der == (1-np.tanh(x._val)**2)*x._der

    #Constant
    assert op.tanh(c) == np.tanh(c)

def test_arcsinh():
    # Scalar
    f = op.arcsinh(x)
    assert f._val == np.arcsinh(x._val)
    assert f._der == (-np.arcsinh(x._val))*np.arctanh(x._val)*x._der

    #Constant
    assert op.arcsinh(c) == np.arcsinh(c)

def test_arccosh():
    # Scalar
    f = op.arccosh(y)
    assert f._val == np.arccosh(y._val)
    assert f._der == -np.arccosh(y._val)*np.tanh(y._val)*y._der

    #Constant
    assert op.arccosh(d) == np.arccosh(d)

def test_arctanh():
    # Scalar
    f = op.arctanh(x)
    assert f._val == np.arctanh(x._val)
    assert f._der == (1-np.arctanh(x._val)**2)*x._der

    #Constant
    assert op.arctanh(c) == np.arctanh(c)

test_sin()
test_cos()
test_tan()
test_arcsin()
test_arccos()
test_arctan()
test_sinh()
test_cosh()
test_tanh()
test_arccosh()
test_arcsinh()
test_arctanh()
test_log()
test_exp()
