#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 17:20:36 2019

@author: weiruchen
"""
import sys
sys.path.append('../')

import numpy as np

from nodes.scalar import Scalar


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
f_7 = f / g
f_8 = g / f
f_9 = f / 2
f_10 = 2 / f


# Power functions
h_1 = f ** 2
h_2 = 2 ** f
h_3 = f ** g



def test_addition():
    """test addition (f+g) for both value and derivative"""
    assert f_1.get()[0] == f.get()[0] + g.get()[0]
    assert f_1.get()[1] == f.get()[1] + g.get()[1]
    """test addition (g+f) for both value and derivative"""
    assert f_2.get()[0] == f.get()[0] + g.get()[0]
    assert f_2.get()[1] == f.get()[1] + g.get()[1]
    """test addition of function and constant for both value and derivative"""
    assert (f_1 + 3.0).get()[0] == f.get()[0] + g.get()[0] + 3.0
    assert (f_1 + 3.0).get()[1] == f.get()[1] + g.get()[1]

def test_subtraction():
    """test subtraction (f-g) for both value and derivative"""
    assert f_3.get()[0] == f.get()[0] - g.get()[0]
    assert f_3.get()[1] == f.get()[1] - g.get()[1]
    """test subtraction (g-f) for both value and derivative"""
    assert f_4.get()[0] == g.get()[0] - f.get()[0]
    assert f_4.get()[1] == g.get()[1] - f.get()[1]
    """test subtraction of constant and function for both value and derivative"""
    assert (3.0 - f_1).get()[0] == 3.0 - f.get()[0] - g.get()[0]
    assert (3.0 - f_1).get()[1] == - f.get()[1] - g.get()[1]

def test_multiplication():
    """test multiplication (f*g) for both value and derivative"""
    assert f_5.get()[0] == f.get()[0] * g.get()[0]
    assert f_5.get()[1] == f.get()[0] * g.get()[1] + f.get()[1] * g.get()[0]
    """test multiplication (g*f) for both value and derivative"""
    assert f_6.get()[0] == g.get()[0] * f.get()[0]
    assert f_6.get()[1] == g.get()[0] * f.get()[1] + g.get()[1] * f.get()[0]
    """test multiplication of constant and function for both value and derivative"""
    assert (3*f).get()[0] == 3*(f.get()[0])
    assert (3*f).get()[1] == 3*(f.get()[1])

def test_pow():
    """test the function to the power of constant for both value and derivative"""
    assert h_1.get()[0] == f.get()[0]**2
    assert h_1.get()[1] == 2*f.get()[0]*f.get()[1]
    """test the constant to the power of function for both value and derivative"""
    assert h_2.get()[0] == 2**f.get()[0]
    assert h_2.get()[1] == 2**f.get()[0]*np.log(2)*f.get()[1]
    """test the value of function to the power of function for both value and derivative"""
    assert h_3.get()[0] == f.get()[0]**g.get()[0]
    assert h_3.get()[1] == np.exp(g.get()[0]*np.log(f.get()[0]))*(g.get()[1]*np.log(f.get()[0])+g.get()[0]/float(f.get()[0]))


def test_divide():
    """test division (f/g) for both value and derivative"""
    assert f_7.get()[0] == f.get()[0] / g.get()[0]
    assert f_7.get()[1] == (f.get()[1]*g.get()[0]-f.get()[0]*g.get()[1]) / g.get()[0]**2
    """test division (g/f) for both value and derivative"""
    assert f_8.get()[0] == g.get()[0] / f.get()[0]
    assert f_8.get()[1] == (g.get()[1]*f.get()[0]-g.get()[0]*f.get()[1]) / f.get()[0]**2
    """test division of function and constant for both value and derivative"""
    assert f_9.get()[0] == f.get()[0] / 2.0
    assert f_9.get()[1] == f.get()[1] / 2.0
    """test division of constant and function for both value and derivative"""
    assert f_10.get()[0] == 2.0 / f.get()[0]
    assert f_10.get()[1] == -2.0 * f.get()[1] / f.get()[0]**2
    

def test_neg():
    """test negation (-f) for both value and derivative"""
    assert (-f).get()[0] == -(f.get()[0])
    assert (-f).get()[1] == -(f.get()[1])
    
test_addition()
test_subtraction()
test_divide()
test_multiplication()
test_pow()
test_neg()


