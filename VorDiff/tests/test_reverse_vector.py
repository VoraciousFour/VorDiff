#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 20:32:48 2019

@author: weiruchen
"""

import numpy as np
from VorDiff.nodes.reverse_vector import ReverseVector


def generate():
    var1, var2 = ReverseVector([10, 10, 3]), ReverseVector([20, 20, 25])
    var1._init_children()
    var2._init_children()
    return var1, var2


# Define constants
c_1, c3, c7 = -1, 3, 7


def test_get_item():
    x = ReverseVector([1, 2])
    assert x[0].get() == 1
    assert x[1].get() == 2

test_get_item()


def test_get():
    x, y = generate()
    assert (x.get() == x._val).all()
    assert (y.get() == y._val).all()

test_get()



def test_add():
    x, y = generate()
    f = x + y 
    assert (f.get() == x._val + y._val).all()
    

    f._gradient = 1
    f.compute_gradient(x)
    f.compute_gradient(y)
    assert (x._gradient == 1).all()
    assert (y._gradient == 1).all()

    x, y = generate()
    g = y + x
    assert (g.get() == x._val + y._val).all()

    g._gradient = 1
    g.compute_gradient(x)
    g.compute_gradient(y)
    assert (x._gradient == 1).all()
    assert (y._gradient == 1).all()

    x, y = generate()
    h = c7 + x + y + c3
    assert (h.get() == c7 + x._val + y._val + c3).all()

    h._gradient = 1
    h.compute_gradient(x)
    h.compute_gradient(y)
    assert (x._gradient == 1).all()
    assert (y._gradient == 1).all()

test_add()




def test_subtract():
    x, y = generate()
    f = x - y
    assert (f.get() == x._val - y._val).all()

    f._gradient = 1
    f.compute_gradient(x)
    f.compute_gradient(y)
    assert (x._gradient == 1).all()
    assert (y._gradient == -1).all()

    x, y = generate()
    g = y - x
    assert (g.get() == y._val - x._val).all()

    g._gradient = 1
    g.compute_gradient(x)
    g.compute_gradient(y)
    assert (x._gradient == -1).all()
    assert (y._gradient == 1).all()

    x, y = generate()
    h = c_1 - x - y - c3
    assert (h.get() == c_1 - x._val - y._val - c3).all()

    h._gradient = 1
    h.compute_gradient(x)
    h.compute_gradient(y)
    assert (x._gradient == -1).all()
    assert (y._gradient == -1).all()

test_subtract()




def test_multiply():
    x, y = generate()
    f = x * y
    assert (f.get() == x._val * y._val).all()

    f._gradient = 1
    f.compute_gradient(x)
    f.compute_gradient(y)
    assert (x._gradient == y._val).all()
    assert (y._gradient == x._val).all()

    x, y = generate()
    g = y * x
    assert (g.get() == x._val * y._val).all()

    g._gradient = 1
    g.compute_gradient(x)
    g.compute_gradient(y)
    assert (x._gradient == y._val).all()
    assert (y._gradient == x._val).all()

    x, y = generate()
    h = c_1 * x * y * c3
    assert (h.get() == c_1 * x._val * y._val * c3).all()

    h._gradient = 1
    h.compute_gradient(x)
    h.compute_gradient(y)
    assert (x._gradient == c_1 * y._val * c3).all()
    assert (y._gradient == c_1 * x._val * c3).all()

test_multiply()






def new_generate():
    vars = ReverseVector([10, 10, 3]), ReverseVector([20, 20, 25]), ReverseVector([30, 30, 1])
    for var in vars:
        var._init_children()
    return tuple(vars)

def test_divide():
    x, y, z = new_generate()
    f = x / y / (1 / z)
    assert (f.get() == x._val / y._val / (1 / z._val)).all()

    f._gradient = 1
    f.compute_gradient(x)
    f.compute_gradient(y)
    f.compute_gradient(z)
    assert (x._gradient == z._val / y._val).all()
    assert (y._gradient == -x._val * z._val / y._val ** 2).all()
    assert (z._gradient == x._val / y._val).all()

    x, y, z = new_generate()
    g = z / (x * y)
    assert (g.get() == z._val / (x._val * y._val)).all()

    g._gradient = 1
    g.compute_gradient(x)
    g.compute_gradient(y)
    g.compute_gradient(z)
    assert (x._gradient == -z._val / (x._val ** 2 * y._val)).all()
    assert (y._gradient == -z._val / (x._val * y._val ** 2)).all()
    assert (z._gradient == 1 / (x._val * y._val)).all()

    x, y, z = new_generate()
    h = c_1 * x / (y * z * c3)
    assert (h.get() == c_1 * x._val / (y._val * z._val * c3)).all()

    h._gradient = 1
    h.compute_gradient(x)
    h.compute_gradient(y)
    h.compute_gradient(z)
    assert (x._gradient == c_1 / (y._val * z._val * c3)).all()
    assert (y._gradient == -c_1 * x._val / (y._val * y._val * z._val * c3)).all()
    assert (z._gradient == -c_1 * x._val / (y._val * z._val ** 2 * c3)).all()

    x, y, z = new_generate()
    w = x / c7
    assert (w.get() == x._val / c7).all()

    w._gradient = 1
    w.compute_gradient(x)
    assert (x._gradient == 1 / c7).all()



test_divide()



def test_power():
    x, y, z = new_generate()
    x._val, y._val, z._val = np.array([2, 2, 2]), np.array([3, 3, 3]), np.array([4, 4, 4])
    f = x ** y * z
    assert (f.get() == x._val ** y._val * z._val).all()

    f._gradient = 1
    f.compute_gradient(x)
    f.compute_gradient(y)
    f.compute_gradient(z)
    assert (x._gradient == y._val * x._val ** (y._val - 1) * z._val).all()
    assert (y._gradient == x._val ** y._val * z._val * np.log(x._val)).all()
    assert (z._gradient == x._val ** y._val).all()

    x, y, z = new_generate()
    g = z ** (x * y)
    assert (g.get() == z._val ** (x._val * y._val)).all()

    g._gradient = 1
    g.compute_gradient(x)
    g.compute_gradient(y)
    g.compute_gradient(z)
    assert (x._gradient == y._val * np.log(z._val) * z._val ** (x._val * y._val)).all()
    assert (y._gradient == x._val * np.log(z._val) * z._val ** (x._val * y._val)).all()
    assert (z._gradient == x._val * y._val * z._val ** (x._val * y._val - 1)).all()

    x, y, z = new_generate()
    x._val, y._val, z._val = np.array([2, 2, 2]), np.array([3, 3, 3]), np.array([4, 4, 4])
    h = c_1 * x ** (y ** z * c3)
    assert (h.get() == c_1 * x._val ** (y._val ** z._val * c3)).all()

    h._gradient = 1
    h.compute_gradient(x)
    h.compute_gradient(y)
    h.compute_gradient(z)
    assert (x._gradient == -c3 * y._val ** z._val * x._val ** (c3 * y._val ** z._val - 1)).all()
    assert (y._gradient == (-c3 * z._val * np.log(x._val) * y._val ** (z._val - 1) * x._val ** (c3 * y._val ** z._val))).all()
    assert (z._gradient == (-c3 * np.log(x._val) * y._val ** z._val * np.log(y._val) * x._val ** (c3 * y._val ** z._val))).all()

    x, y, z = new_generate()
    q = 3 ** x
    assert (q.get() == 3 ** x._val).all()

    q._gradient = 1
    q.compute_gradient(x)
    assert (x._gradient == 3 ** x._val * np.log(3)).all()

    
    x, y, z = new_generate()
    w = x ** 2
    assert (w.get() == x._val ** 2).all()

    w._gradient = 1
    w.compute_gradient(x)
    assert (x._gradient == 2 * x._val).all()



test_power()

