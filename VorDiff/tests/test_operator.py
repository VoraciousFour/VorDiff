import sys
sys.path.append('..')

from VorDiff.operator import Operator as op
from VorDiff.autodiff import AutoDiff as ad

import numpy as np
import math

x = ad.scalar(0.5)
c = 0.5
y = ad.scalar(2.0)
d = 2.0
#z is a two variable function z = x1+x2 at x1 = 1 and x2 = 1 
z = ad.element(2.0, [1.0,1.0]) 
k = ad.element(0.5, [1.0,1.0])

def test_sin():
    
    # Scalar
    f = op.sin(x)
    assert f._val == np.sin(x._val)
    assert f._der == np.cos(x._val)*x._der

    #Constant
    assert op.sin(c) == np.sin(c)
    
    #element
    g = op.sin(z)
    assert g._val == np.sin(z._val)
    assert (g._jacob == np.cos(z._val)*z._jacob).all()

def test_cos():
    
    # Scalar
    f = op.cos(x)
    assert f._val == np.cos(x._val)
    assert f._der == -np.sin(x._val)*x._der

    #Constant
    assert op.cos(c) == np.cos(c)

    #element
    g = op.cos(z)
    assert g._val == np.cos(z._val)
    assert (g._jacob == -np.sin(z._val)*z._jacob).all()
    
def test_tan():
    
    # Scalar
    f = op.tan(x)
    assert f._val == np.tan(x._val)
    assert f._der == x._der/np.cos(x._val)**2

    #Constant
    assert op.tan(c) == np.tan(c)

    #element
    g = op.tan(z)
    assert g._val == np.tan(z._val)
    assert (g._jacob == z._jacob/np.cos(z._val)**2).all()
    
def test_arcsin():
    
    # Scalar
    f = op.arcsin(x)
    assert f._val == np.arcsin(x._val)
    assert f._der == 1/(x._der*(1-x._val**2)**.5)
    
    #Constant
    assert op.arcsin(c) == np.arcsin(c)

    #element
    try:
        g = op.arcsin(z)
    except:
        g = op.arcsin(k)
        assert g._val == np.arcsin(k._val)
        assert (g._jacob == 1/(k._jacob*(1-k._val**2)**.5)).all()
    
def test_arccos():
    
    # Scalar
    f = op.arccos(x)
    assert f._val == np.arccos(x._val)
    assert f._der == -x._der/(1-x._val**2)**.5

    #Constant
    assert op.arccos(c) == np.arccos(c)

    #element
    try:
        g = op.arccos(z)
    except:
        g = op.arccos(k)
        assert g._val == np.arccos(k._val)
        assert (g._jacob == -k._jacob/(1-k._val**2)**.5).all()
    
def test_arctan():
    
    # Scalar
    f = op.arctan(x)
    assert f._val == np.arctan(x._val)
    assert f._der == x._der/(1+x._val**2)

    #Constant
    assert op.arctan(c) == np.arctan(c)
    
    #element

    g = op.arctan(z)
    assert g._val == np.arctan(z._val)
    assert (g._jacob == z._jacob/(1+z._val**2)).all()

def test_log():

    # Scalar
    f = op.log(x)
    assert f._val == np.log(x._val)
    assert f._der == x._der/x._val

    #Constant
    assert op.log(c) == np.log(c)

    #element
    g = op.log(z)
    assert g._val == np.log(z._val)
    assert (g._jacob == z._jacob/z._val).all()

    h = op.log(2,z)
    assert g._val == math.log(2,z._val)
    assert (g._jacob == z._jacob/(z._val*np.log(2))).all()

def test_exp():
    
    # Scalar
    f = op.exp(x)
    assert f._val == np.exp(x._val)
    assert f._der == x._der*np.exp(x._val)

    #Constant
    assert op.exp(c) == np.exp(c)
    
    #element
    g = op.exp(z)
    assert g._val == np.exp(z._val)
    assert (g._jacob == z._jacob*(np.exp(z._val))).all()

    h = op.exp(2,z)
    assert g._val == 2**z._val
    assert (g._jacob == z._jacob*np.log(a)*(a**z._val)).all()

def test_sinh():
    
    # Scalar
    f = op.sinh(x)
    assert f._val == np.sinh(x._val)
    assert f._der == np.cosh(x._val)*x._der

    #Constant
    assert op.sinh(c) == np.sinh(c)

    #element
    g = op.sinh(z)
    assert g._val == np.sinh(z._val)
    assert (g._jacob == z._jacob*(np.cosh(z._val))).all()


def test_cosh():

    # Scalar
    f = op.cosh(x)
    assert f._val == np.cosh(x._val)
    assert f._der == np.sinh(x._val)*x._der

    #Constant
    assert op.cosh(c) == np.cosh(c)

    #element
    g = op.cosh(k)
    assert g._val == np.cosh(k._val)
    assert (g._jacob == z._jacob*(np.sinh(k._val))).all()

def test_tanh():

    # Scalar
    f = op.tanh(x)
    assert f._val == np.tanh(x._val)
    assert f._der == (1-np.tanh(x._val)**2)*x._der
    
    #Constant
    assert op.tanh(c) == np.tanh(c)

    #element
    g = op.tanh(z)
    assert g._val == np.tanh(z._val)
    assert (g._jacob == z._jacob*(1-np.tanh(z._val)**2)).all()

def test_arcsinh():

    # Scalar
    f = op.arcsinh(x)
    assert f._val == np.arcsinh(x._val)
    assert f._der == (-np.arcsinh(x._val))*np.arctanh(x._val)*x._der

    #Constant
    assert op.arcsinh(c) == np.arcsinh(c)

    #element
    g = op.arcsinh(z)
    assert g._val == np.arcsinh(z._val)
    assert (g._jacob == z._jacob*(-np.arcsinh(z._val)*np.arctanh(z._val))).all()


def test_arccosh():

    # Scalar
    f = op.arccosh(y)
    assert f._val == np.arccosh(y._val)
    assert f._der == -np.arccosh(y._val)*np.tanh(y._val)*y._der

    #Constant
    assert op.arccosh(d) == np.arccosh(d)

    #element
    g = op.arccosh(z)
    assert g._val == np.arccosh(z._val)
    assert (g._jacob == z._jacob*(-np.arccosh(z._val)*np.tanh(z._val))).all()

def test_arctanh():

    # Scalar
    f = op.arctanh(x)
    assert f._val == np.arctanh(x._val)
    assert f._der == (1-np.arctanh(x._val)**2)*x._der

    #Constant
    assert op.arctanh(c) == np.arctanh(c)

    #element
    g = op.arctanh(k)
    assert g._val == np.arctanh(k._val)
    assert (g._jacob == k._jacob*(1-np.arctanh(k._val)**2)).all()

def test_logistic():

    #scalar
    f = op.logistic(x)
    assert f._val == 1/(1+np.exp(-x._val))
    assert f._der == x._der*(x._val**2*np.exp(-x._val))

    #constant
    assert op.logistic(c) == 1/(1+np.exp(-c))

    #element
    g = op.logistic(z)
    assert g._val == 1/(1+np.exp(-z._val))
    assert (g._jacob == z._jacob*(z._val**2*np.exp(-z._val))).all()


def test_square_root():

    #scalar
    f = op.square_root(x)
    assert f._val == x._val**(-0.5)
    assert f._der == x._der*(x._val**(-1/2)/2)

    #constant
    assert op.square_root(c) == c**0.5

    #element
    g = op.square_root(z)
    assert g._val == z._val**0.5
    assert (g._jacob == z._jacob*(z._val**(-1/2)/2)).all()


def test_log_():

    #scalar
    f = op.log_(2,x)
    assert f._val == math.log(2, x._val)
    assert f._der == x._der/(x._val*np.log(2))

    #constant
    assert op.log_(2,c) == math.log(2,c)

    #element
    h = op.log_(2,z)
    assert g._val == math.log(2,z._val)
    assert (g._jacob == z._jacob/(z._val*np.log(2))).all()


def test_exp_():

    #scalar
    f = op.exp_(2,x)
    assert f._val == 2**x._val
    assert f._der == x._der*(2**x._val)*np.log(2)

    #constant
    assert op.exp_(2,c) == 2**c

    #element
    h = op.exp_(2,z)
    assert g._val == 2**z._val
    assert (g._jacob == z._jacob*np.log(a)*(a**z._val)).all()



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
test_logistic()
test_exp_()
test_square_root()
test_log_()
