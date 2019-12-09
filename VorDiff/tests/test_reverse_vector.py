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
    var1, var2 = ReverseVector([10, 10, 3]), ReverseVector([20, 20, 25])
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
    x, y = create()
    f = x + y 
    assert (f.get() == x._val + y._val).all()
    

    f._gradient = 1
    f.compute_gradient(x)
    f.compute_gradient(y)
    assert (x._gradient == 1).all()
    assert (y._gradient == 1).all()

    x, y = create()
    f = y + x
    assert (f.get() == x._val + y._val).all()

    f._gradient = 1
    f.compute_gradient(x)
    f.compute_gradient(y)
    assert (x._gradient == 1).all()
    assert (y._gradient == 1).all()

    x, y = create()
    f = c7 + x + y + c3
    assert (f.get() == c7 + x._val + y._val + c3).all()

    f._gradient = 1
    f.compute_gradient(x)
    f.compute_gradient(y)
    assert (x._gradient == 1).all()
    assert (y._gradient == 1).all()

test_addition()




def test_subtraction():
    x, y = create()
    f = x - y
    assert (f.get() == x._val - y._val).all()

    f._gradient = 1
    f.compute_gradient(x)
    f.compute_gradient(y)
    assert (x._gradient == 1).all()
    assert (y._gradient == -1).all()

    x, y = create()
    f = y - x
    assert (f.get() == y._val - x._val).all()

    f._gradient = 1
    f.compute_gradient(x)
    f.compute_gradient(y)
    assert (x._gradient == -1).all()
    assert (y._gradient == 1).all()

    x, y = create()
    f = c_1 - x - y - c3
    assert (f.get() == c_1 - x._val - y._val - c3).all()

    f._gradient = 1
    f.compute_gradient(x)
    f.compute_gradient(y)
    assert (x._gradient == -1).all()
    assert (y._gradient == -1).all()

test_subtraction()




def test_multiplication():
    x, y = create()
    f = x * y
    assert (f.get() == x._val * y._val).all()

    f._gradient = 1
    f.compute_gradient(x)
    f.compute_gradient(y)
    assert (x._gradient == y._val).all()
    assert (y._gradient == x._val).all()

    x, y = create()
    f = y * x
    assert (f.get() == x._val * y._val).all()

    f._gradient = 1
    f.compute_gradient(x)
    f.compute_gradient(y)
    assert (x._gradient == y._val).all()
    assert (y._gradient == x._val).all()

    x, y = create()
    f = c_1 * x * y * c3
    assert (f.get() == c_1 * x._val * y._val * c3).all()

    f._gradient = 1
    f.compute_gradient(x)
    f.compute_gradient(y)
    assert (x._gradient == c_1 * y._val * c3).all()
    assert (y._gradient == c_1 * x._val * c3).all()

test_multiplication()






def new_create():
    vars = ReverseVector([10, 10, 3]), ReverseVector([20, 20, 25]), ReverseVector([30, 30, 1])
    for var in vars:
        var._init_children()
    return tuple(vars)

def test_division():
    x, y, z = new_create()
    f = x / y / (1 / z)
    assert (f.get() == x._val / y._val / (1 / z._val)).all()

    f._gradient = 1
    f.compute_gradient(x)
    f.compute_gradient(y)
    f.compute_gradient(z)
    assert (x._gradient == z._val / y._val).all()
    assert (y._gradient == -x._val * z._val / y._val ** 2).all()
    assert (z._gradient == x._val / y._val).all()

    x, y, z = new_create()
    f = z / (x * y)
    assert (f.get() == z._val / (x._val * y._val)).all()

    f._gradient = 1
    f.compute_gradient(x)
    f.compute_gradient(y)
    f.compute_gradient(z)
    assert (x._gradient == -z._val / (x._val ** 2 * y._val)).all()
    assert (y._gradient == -z._val / (x._val * y._val ** 2)).all()
    assert (z._gradient == 1 / (x._val * y._val)).all()

    x, y, z = new_create()
    f = c_1 * x / (y * z * c3)
    assert (f.get() == c_1 * x._val / (y._val * z._val * c3)).all()

    f._gradient = 1
    f.compute_gradient(x)
    f.compute_gradient(y)
    f.compute_gradient(z)
    assert (x._gradient == c_1 / (y._val * z._val * c3)).all()
    assert (y._gradient == -c_1 * x._val / (y._val * y._val * z._val * c3)).all()
    assert (z._gradient == -c_1 * x._val / (y._val * z._val ** 2 * c3)).all()

    x, y, z = new_create()
    f = x / c7
    assert (f.get() == x._val / c7).all()

    f._gradient = 1
    f.compute_gradient(x)
    assert (x._gradient == 1 / c7).all()



test_division()



def test_power():
    x, y, z = new_create()
    x._val, y._val, z._val = np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3])
    f = x ** y * z
    assert (f.get() == x._val ** y._val * z._val).all()

    f._gradient = 1
    f.compute_gradient(x)
    f.compute_gradient(y)
    f.compute_gradient(z)
    assert (x._gradient == y._val * x._val ** (y._val - 1) * z._val).all()
    assert (y._gradient == x._val ** y._val * z._val * np.log(x._val)).all()
    assert (z._gradient == x._val ** y._val).all()

#    x, y, z = new_create()
#    f = z ** (x * y)
#    assert (f.get() == z._val ** (x._val * y._val)).all()
#
#    f._gradient = 1
#    f.compute_gradient(x)
#    f.compute_gradient(y)
#    f.compute_gradient(z)
#    assert (x._gradient == y._val * np.log(z._val) * z._val ** (x._val * y._val)).all()
#    assert (y._gradient == x._val * np.log(z._val) * z._val ** (x._val * y._val)).all()
#    assert (z._gradient == x._val * y._val * z._val ** (x._val * y._val - 1)).all()

    x, y, z = new_create()
    x._val, y._val, z._val = np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([3, 3, 3])
    f = c_1 * x ** (y ** z * c3)
    assert (f.get() == c_1 * x._val ** (y._val ** z._val * c3)).all()

    f._gradient = 1
    f.compute_gradient(x)
    f.compute_gradient(y)
    f.compute_gradient(z)
    assert (x._gradient == -c3 * y._val ** z._val * x._val ** (c3 * y._val ** z._val - 1)).all()
    assert (y._gradient == (-c3 * z._val * np.log(x._val) * y._val ** (z._val - 1) * x._val ** (c3 * y._val ** z._val))).all()
    assert (z._gradient == (-c3 * np.log(x._val) * y._val ** z._val * np.log(y._val) * x._val ** (c3 * y._val ** z._val))).all()

    x, y, z = new_create()
    f = 3 ** x
    assert (f.get() == 3 ** x._val).all()

    f._gradient = 1
    f.compute_gradient(x)
    assert (x._gradient == 3 ** x._val * np.log(3)).all()

    
    x, y, z = new_create()
    f = x ** 2
    assert (f.get() == x._val ** 2).all()

    f._gradient = 1
    f.compute_gradient(x)
    assert (x._gradient == 2 * x._val).all()



test_power()

