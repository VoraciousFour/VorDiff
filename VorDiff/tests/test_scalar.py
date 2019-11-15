#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 17:20:36 2019

@author: weiruchen
"""

from scalar import Scalar

# Define scalar object and initialize parameters

a = 2.0
b = 4.0
c = 1.0
d = 3.0


x = Scalar(a)


f = a * x + b
g = c * x + d


# Define functions of the scalar objects
f_1 = f + g
f_2 = g + f
f_3 = f - g
f_4 = g - f
f_5 = f * g
f_6 = g * f
f_7 = f / Scalar(5)
f_8 = Scalar(5) / f

# Power functions
h_1 = f ** 2
h_2 = g ** Scalar(3)


def test_addition():
    assert f_1.get()[0] == f.get()[0] + g.get()[0]
    assert f_1.get()[1] == f.get()[1] + g.get()[1]
    assert f_2.get()[0] == f.get()[0] + g.get()[0]
    assert f_2.get()[1] == f.get()[1] + g.get()[1]
    assert (f_1 + 3.0).get()[0] == f.get()[0] + g.get()[0] + 3.0
    assert (f_1 + 3.0).get()[1] == f.get()[1] + g.get()[1]

def test_subtraction():
    assert f_3.get()[0] == f.get()[0] - g.get()[0]
    assert f_3.get()[1] == f.get()[1] - g.get()[1]
    assert f_4.get()[0] == g.get()[0] - f.get()[0]
    assert f_4.get()[1] == g.get()[1] - f.get()[1]
    assert (3.0 - f_1).get()[0] == 3.0 - f.get()[0] - g.get()[0]
    assert (3.0 - f_1).get()[1] == - f.get()[1] - g.get()[1]

def test_multiplication():
    assert f_5.get()[0] == f.get()[0] * g.get()[0]
    assert f_5.get()[1] == f.get()[0] * g.get()[1] + f.get()[1] * g.get()[0]
    assert f_6.get()[0] == g.get()[0] * f.get()[0]
    assert f_6.get()[1] == g.get()[0] * f.get()[1] + g.get()[1] * f.get()[0]

def test_pow():
    assert h_1.get()[0] == f.get()[0]**2
    assert h_1.get()[1] == 2*f.get()[0]*f.get()[1]
    assert h_2.get()[0] == g.get()[0]**3
    assert h_2.get()[1] == 3*(g.get()[0]**2)*g.get()[1]
"""
def test_divide():
    assert f_7.get()[0] == f.get()[0] / Scalar(5).get()[0]
    assert f_7.get()[1] == f.get()[1] / Scalar(5).get()[1]
    assert f_8.get()[0] == Scalar(5).get()[0] / f.get()[0]
    assert f_8.get()[1] == Scalar(5).get()[1] / f.get()[1]
test_addition()
test_subtraction()
test_divide()
print(f_8.get())
print(f.get()[0] / Scalar(5).get()[0])
test_multiplication()
test_pow()
"""