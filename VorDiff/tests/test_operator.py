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

def test_sinh():
    # Scalar
    f = op.sinh(x)
    assert f._val == np.sinh(x._val)
    assert f._der == np.cosh(x._val)*x_der

    #Constant
    assert op.sinh(c) == np.sinh(c)

def test_cosh():
    # Scalar
    f = op.cosh(x)
    assert f._val == np.cosh(x._val)
    assert f._der == np.sinh(x._val)*x_der

    #Constant
    assert op.cosh(c) == np.cosh(c)

def test_tanh():
    # Scalar
    f = op.tanh(x)
    assert f._val == np.tanh(x._val)
    assert f._der == (1-np.tanh(x._val)**2)*x_der

    #Constant
    assert op.tanh(c) == np.tanh(c)

def test_arcsinh():
    # Scalar
    f = op.arcsinh()
    assert f._val == np.arcsinh(x._val)
    assert f._der == (-np.arcsinh(x._val))*np.arctanh(x._val)*x_der

    #Constant
    assert op.arcsinh(c) == np.arcsinh(c)

def test_arccosh():
    # Scalar
    f = op.arccosh(x)
    assert f._val == np.arccosh(x._val)
    assert f._der == -np.arccosh(x._val)*np.tanh(x._val)*x_der

    #Constant
    assert op.arccosh(c) == np.arccosh(c)

def test_arctanh():
    # Scalar
    f = op.arctanh(x)
    assert f._val == np.arctanh(x._val)
    assert f._der == (1-np.arctanh(x._val)**2)*x_der

    #Constant
    assert op.arctanh(c) == np.arctanh(c)

