from VorDiff.nodes.vector import Vector
from VorDiff.autodiff import AutoDiff as ad
import numpy as np

x = ad.vector([1,2, 3, 4])
a = 6
b = 2

x1 = x[0]
x2 = x[1]
x3 = x[2]
x4 = x[3]

f1 = x1+x2
f2 = x2+x1
f3 = x1-x2
f4 = x2-x1
f5 = x3*x4
f6 = x4*x3
f7 = x1/x2
f8 = x2/x1

# Power functions
h1 = f1 ** b
h2 = b ** f1
h3 = x2 ** x3
h4 = x3 ** x2

# Equal or not equal
e1 = x1+x3
e2 = x3+x1
e3 = x2+x4
e4 = x4+x2


def test_addition():
    """test addition (x1+x2) for both value and derivative"""
    assert f1.get_val() == x1.get_val()+x2.get_val()
    assert f1.get_derivatives().all() == (x1.get_derivatives()+x2.get_derivatives()).all()
    """test addition (x2+x1) for both value and derivative"""
    assert f2.get_val() == x1.get_val()+x2.get_val()
    assert f2.get_derivatives().all() == (x1.get_derivatives()+x2.get_derivatives()).all()
    """test addition of function and constant for both value and derivative"""
    assert (f1+a).get_val() == f1.get_val()+a
    assert (f1+a).get_derivatives().all() == f1.get_derivatives().all()
    """test addition of constant and function for both value and derivative"""
    assert (a+f1).get_val() == f1.get_val()+a
    assert (a+f1).get_derivatives().all() == f1.get_derivatives().all()

test_addition()

def test_subtraction():
    """test subtraction (x1-x2) for both value and derivative"""
    assert f3.get_val() == x1.get_val()-x2.get_val()
    assert f3.get_derivatives().all() == (x1.get_derivatives()-x2.get_derivatives()).all()
    """test subtraction (x2-x1) for both value and derivative"""
    assert f4.get_val() == x2.get_val()-x1.get_val()
    assert f4.get_derivatives().all() == (x2.get_derivatives()-x1.get_derivatives()).all()
    """test subtraction of function and constant for both value and derivative"""
    assert (f1-a).get_val() == f1.get_val()-a
    assert (f1-a).get_derivatives().all() == f1.get_derivatives().all()
    """test subtraction of constant and function for both value and derivative"""
    assert (a-f1).get_val() == a-f1.get_val()
    assert (a-f1).get_derivatives().all() == (-f1.get_derivatives()).all()

test_subtraction()

def test_multiplication():
    """test multiplication (x3*x4) for both value and derivative"""
    assert f5.get_val() == x3.get_val()*x4.get_val()
    assert f5.get_derivatives().all() == (np.array([0,0,1,1])).all()
    """test multiplication (x4*x3) for both value and derivative"""
    assert f6.get_val() == x3.get_val()*x4.get_val()
    assert f6.get_derivatives().all() == (np.array([0,0,1,1])).all()
    """test multiplication of function and constant for both value and derivative"""
    assert (f2*a).get_val() == f2.get_val()*a
    assert ((f2*a).get_derivatives() == np.array([a, a, 0, 0])).all()
    """test multiplication of constant and function for both value and derivative"""
    assert (a*f3).get_val() == f3.get_val()*a
    assert ((a*f3).get_derivatives() == np.array([a, -a, 0, 0])).all()

test_multiplication()

def test_divide():
    """test division (x1/x2) for both value and derivative"""
    assert f7.get_val() == x1.get_val()/x2.get_val()
    assert f7.get_derivatives().all() == (np.array([1/2,-1/4,0,0])).all()
    """test division (x2/x1) for both value and derivative"""
    assert f8.get_val() == x2.get_val()/x1.get_val()
    assert f8.get_derivatives().all() == (np.array([-2,1,0,0])).all()
    """test division of function and constant for both value and derivative"""
    assert (x2/a).get_val() == x2.get_val()/a
    assert ((x2/a).get_derivatives() == np.array([0, 1/a, 0, 0])).all()
    """test division of constant and function for both value and derivative"""
    assert (a/x3).get_val() == a/x3.get_val()
    assert ((a/x3).get_derivatives() == np.array([0, 0, -a/9, 0])).all()

test_divide()

def test_pow():
    """test the function to the power of constant for both value and derivative"""
    assert h1.get_val() == f1.get_val()**b
    assert (h1.get_derivatives() == np.array([6, 6, 0, 0])).all()
    """test the constant to the power of function for both value and derivative"""
    assert h2.get_val() == b**f1.get_val()
    assert (h2.get_derivatives() == np.array([8*np.log(2), 8*np.log(2), 0, 0])).all()
    """test the value of function to the power of function (x2**x3) for both value and derivative"""
    assert h3.get_val() == x2.get_val()**x3.get_val()
    assert (np.round(h3.get_derivatives(),decimals=3) == np.round(np.array([0,8*3/2,8*np.log(2),0]),decimals=3)).all()
    """test the value of function to the power of function (x3**x2) for both value and derivative"""
    assert h4.get_val() == x3.get_val()**x2.get_val()
    assert (np.round(h4.get_derivatives(),decimals=3) == np.round(np.array([0,9*np.log(3),9*2/3,0]),decimals=3)).all()

test_pow()

def test_neg():
    """test negation (-f1) for both value and derivative"""
    assert (-f1).get_val() == -(f1.get_val())
    assert ((-f1).get_derivatives() == -(f1.get_derivatives())).all()
    
test_neg()

def test_equal():
    """test if two functions (e1, e2) are equal"""
    assert (e1 == e2) == True
    """test if two functions (e1, e3) are equal"""
    assert (e1 == e3) == False
    """test if a function and a constant (e1, a) are equal"""
    assert (e1 == a) == False
    
test_equal()

def test_notequal():
    """test if two functions (e3, e4) are not equal"""
    assert (e3 != e4) == False
    """test if two functions (e2, e4) are not equal"""
    assert (e2 != e4) == True
    """test if a function and a constant (e1, a) are not equal"""
    assert (e1 != a) == True

test_notequal()



