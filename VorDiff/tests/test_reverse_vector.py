#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 20:32:48 2019

@author: weiruchen
"""

import numpy as np
from VorDiff.nodes.reverse_vector import ReverseVector

# create reverse vectors
def create():
    var1, var2 = ReverseVector([10, 3]), ReverseVector([20, 25])
    var1._init_children()
    var2._init_children()
    return var1, var2


# Define constants
c_1, c3, c7 = -1, 3, 7


def test_getitem():
    x = ReverseVector([1, 2])
    assert x[0].get() == 1
    assert x[1].get() == 2

test_getitem()


def test_get():
    x, y = create()
    assert (x.get() == x._val).all()
    assert (y.get() == y._val).all()

test_get()



def test_addition():
    """test addition for self + other: (x+y)"""
    x, y = create()
    f = x + y 
    assert (f.get() == x._val + y._val).all()
    

    f._gradient = 1
    f.compute_gradient(x)
    f.compute_gradient(y)
    assert (x._gradient == 1).all()
    assert (y._gradient == 1).all()
    
    
    """test addition for other + self: (y+x)"""

    x, y = create()
    f = y + x
    assert (f.get() == x._val + y._val).all()

    f._gradient = 1
    f.compute_gradient(x)
    f.compute_gradient(y)
    assert (x._gradient == 1).all()
    assert (y._gradient == 1).all()
    
    """test addition for constant + self"""

    x, y = create()
    f = c7 + x
    assert (f.get() == c7 + x._val).all()

    f._gradient = 1
    f.compute_gradient(x)
    assert (x._gradient == 1).all()
    
    """test addition for self + constant"""
    
    x, y = create()
    f = x + c3
    assert (f.get() == x._val + c3).all()

    f._gradient = 1
    f.compute_gradient(x)
    assert (x._gradient == 1).all()

test_addition()




def test_subtraction():
    """test subtraction for self - other: (x-y)"""
    x, y = create()
    f = x - y
    assert (f.get() == x._val - y._val).all()

    f._gradient = 1
    f.compute_gradient(x)
    f.compute_gradient(y)
    assert (x._gradient == 1).all()
    assert (y._gradient == -1).all()

    """test subtraction for other - self: (y - x)"""
    
    x, y = create()
    f = y - x
    assert (f.get() == y._val - x._val).all()

    f._gradient = 1
    f.compute_gradient(x)
    f.compute_gradient(y)
    assert (x._gradient == -1).all()
    assert (y._gradient == 1).all()
    
    """test subtraction for constant - self"""

    x, y = create()
    f = c_1 - x
    assert (f.get() == c_1 - x._val).all()

    f._gradient = 1
    f.compute_gradient(x)
    assert (x._gradient == -1).all()
    
    """test subtraction for self - constant"""
    
    x, y = create()
    f = - x - c3
    assert (f.get() ==  - x._val - c3).all()

    f._gradient = 1
    f.compute_gradient(x)
    assert (x._gradient == -1).all()

test_subtraction()




def test_multiplication():
    """test multiplication for self * other: (x*y)"""
    x, y = create()
    f = x * y
    assert (f.get() == x._val * y._val).all()

    f._gradient = 1
    f.compute_gradient(x)
    f.compute_gradient(y)
    assert (x._gradient == y._val).all()
    assert (y._gradient == x._val).all()
    
    """test multiplication for other * self: (y*x)"""

    x, y = create()
    f = y * x
    assert (f.get() == x._val * y._val).all()

    f._gradient = 1
    f.compute_gradient(x)
    f.compute_gradient(y)
    assert (x._gradient == y._val).all()
    assert (y._gradient == x._val).all()
    
    
    """test multiplication for constant * self: (c_1 * x)"""

    x, y = create()
    f = c_1 * x
    assert (f.get() == c_1 * x._val).all()

    f._gradient = 1
    f.compute_gradient(x)
    assert (x._gradient == c_1).all()
    
    
    """test multiplication for self * constant: (x * c3)"""

    x, y = create()
    f = x * c3
    assert (f.get() == x._val * c3).all()

    f._gradient = 1
    f.compute_gradient(x)
    assert (x._gradient == c3).all()

test_multiplication()




def new_create():
    vars = ReverseVector([10, 10, 3]), ReverseVector([20, 20, 25]), ReverseVector([30, 30, 1])
    for var in vars:
        var._init_children()
    return tuple(vars)

#def test_division():
#    """test division for (x/y/(1/z))"""
#    x, y, z = new_create()
#    f = x / y / (1 / z)
#    assert (f.get() == x._val / y._val / (1 / z._val)).all()
#
#    f._gradient = 1
#    f.compute_gradient(x)
#    f.compute_gradient(y)
#    f.compute_gradient(z)
#    assert (x._gradient == z._val / y._val).all()
#    assert (y._gradient == -x._val * z._val / y._val ** 2).all()
#    assert (z._gradient == x._val / y._val).all()
#
#    x, y, z = new_create()
#    f = z / (x * y)
#    assert (f.get() == z._val / (x._val * y._val)).all()
#
#    f._gradient = 1
#    f.compute_gradient(x)
#    f.compute_gradient(y)
#    f.compute_gradient(z)
#    assert (x._gradient == -z._val / (x._val ** 2 * y._val)).all()
#    assert (y._gradient == -z._val / (x._val * y._val ** 2)).all()
#    assert (z._gradient == 1 / (x._val * y._val)).all()
#
#    x, y, z = new_create()
#    f = c_1 * x / (y * z * c3)
#    assert (f.get() == c_1 * x._val / (y._val * z._val * c3)).all()
#
#    f._gradient = 1
#    f.compute_gradient(x)
#    f.compute_gradient(y)
#    f.compute_gradient(z)
#    assert (x._gradient == c_1 / (y._val * z._val * c3)).all()
#    assert (y._gradient == -c_1 * x._val / (y._val * y._val * z._val * c3)).all()
#    assert (z._gradient == -c_1 * x._val / (y._val * z._val ** 2 * c3)).all()
#
#    x, y, z = new_create()
#    f = x / c7
#    assert (f.get() == x._val / c7).all()
#
#    f._gradient = 1
#    f.compute_gradient(x)
#    assert (x._gradient == 1 / c7).all()
#
#
#
#test_division()

def test_division():
    """test division for self/other: (x/y)"""
    x, y, z = new_create()
    f = x / y
    assert (f.get() == x._val / y._val).all()

    f._gradient = 1
    f.compute_gradient(x)
    f.compute_gradient(y)
    assert (x._gradient == 1 / y._val).all()
    assert (y._gradient == -x._val / y._val ** 2).all()
    
    """test division for other/self: (z/x)"""

    x, y, z = new_create()
    f = z / x
    assert (f.get() == z._val / x._val).all()

    f._gradient = 1
    f.compute_gradient(x)
    f.compute_gradient(z)
    assert (x._gradient == -z._val / (x._val ** 2)).all()
    assert (z._gradient == 1 / (x._val)).all()
    
    
    """test division for constant/self: (c_1/y)"""

    x, y, z = new_create()
    f = c_1 / y
    assert (f.get() == c_1 / (y._val)).all()

    f._gradient = 1
    f.compute_gradient(y)
    assert (y._gradient == -c_1 / (y._val ** 2)).all()
    
    """test division for self/constant: (x/c7)"""

    x, y, z = new_create()
    f = x / c7
    assert (f.get() == x._val / c7).all()

    f._gradient = 1
    f.compute_gradient(x)
    assert (x._gradient == 1 / c7).all()



test_division()



def test_power():
    """test power for self**other: (x**y)"""
    x, y, z = new_create()
    x._val, y._val, z._val = np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3])
    f = x ** y
    assert (f.get() == x._val ** y._val).all()

    f._gradient = 1
    f.compute_gradient(x)
    f.compute_gradient(y)
    assert (x._gradient == y._val * x._val ** (y._val - 1)).all()
    assert (y._gradient == x._val ** y._val * np.log(x._val)).all()
   
#    """test power for self**other: (x**y)"""
#
#    x, y, z = new_create()
#    x._val, y._val, z._val = np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3])
#    f = c_1 * x ** (y ** z * c3)
#    assert (f.get() == c_1 * x._val ** (y._val ** z._val * c3)).all()
#
#    f._gradient = 1
#    f.compute_gradient(x)
#    f.compute_gradient(y)
#    f.compute_gradient(z)
#    assert (x._gradient == -c3 * y._val ** z._val * x._val ** (c3 * y._val ** z._val - 1)).all()
#    assert (y._gradient == (-c3 * z._val * np.log(x._val) * y._val ** (z._val - 1) * x._val ** (c3 * y._val ** z._val))).all()
#    assert (z._gradient == (-c3 * np.log(x._val) * y._val ** z._val * np.log(y._val) * x._val ** (c3 * y._val ** z._val))).all()

    """test power for constant ** self"""
    x, y, z = new_create()
    f = 3 ** x
    assert (f.get() == 3 ** x._val).all()

    f._gradient = 1
    f.compute_gradient(x)
    assert (x._gradient == 3 ** x._val * np.log(3)).all()

    """test power for self ** constant"""
    
    x, y, z = new_create()
    f = x ** 2
    assert (f.get() == x._val ** 2).all()

    f._gradient = 1
    f.compute_gradient(x)
    assert (x._gradient == 2 * x._val).all()



test_power()
