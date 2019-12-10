#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 17:09:08 2019

@author: weiruchen
"""

from VorDiff.operator import Operator as op
from VorDiff.autodiff import AutoDiff as ad
#from operator_file import Operator as op
#from autodiff import AutoDiff as ad

x = ad.vector([2, 1, 0, 3, 6])
y = ad.vector([-1, 2, 1])

def f1(a, b):
    return 2*a + b

def f2(a):
    return a**2

def f3(a, b):
    return b/a

def f4(a, b, c):
    return op.sin(a+2*b-c)

def f5(a):
    return op.exp(a)

# vector functions

def F1(x):
    x1, x2, x3, x4, x5 = x
    return f5(f4(x3,f3(f2(f1(x1,x5)),x2),x4))

def F2(y):
    y1, y2, y3 = y
    return f1(f2(f4(y1, y2, y3)),y3)

vec_functions = [F1(x), F2(y)]
vals = []
derivatives = []

for function in vec_functions:
    vals.append(function.get_val())
    derivatives.append(function.get_derivatives())

print(vals)
print(derivatives)



#sequence of functions
x1, x2, x3, x4, x5 = x
e1 = f1(x1, x5)
e2 = f2(e1)
e3 = f3(e2, x2)
e4 = f4(x3, e3, x4)
e5 = f5(e4)

print(e1.get_val())
print(e2.get_val())
print(e3.get_val())
print(e4.get_val())
print(e5.get_val())

print(e1.get_derivatives())
print(e2.get_derivatives())
print(e3.get_derivatives())
print(e4.get_derivatives())
print(e5.get_derivatives())

