#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:34:04 2019

@author: weiruchen
"""

from VorDiff.reverse_operator import ReverseOperator as rop
from VorDiff.reverse_autodiff import ReverseAutoDiff as rad


def create_reverse_vector(array):
    x, y = rad.reverse_vector(array)
    return x, y

x,y = create_reverse_vector([[1, 2, 3], [1,3,6]])

# for scalar
f = 1 / (x[1]) + rop.sin(1/x[1])
print(rad.partial_scalar(f))


# for vector
x,y = create_reverse_vector([[1, 2, 3], [1,3,6]])

a = x + 1
print(rad.partial_vector(a,x))


x,y = create_reverse_vector([[1, 2, 3], [1,3,6]])

h = rop.sin(x)
print(rad.partial_vector(h,x))




x,y = create_reverse_vector([[1, 2, 3], [1,3,6]])

g = rop.cos(y)**2
print(rad.partial_vector(g,y))

x,y = create_reverse_vector([[1, 2, 3], [1,3,6]])

f = 2*x + y
print(rad.partial_vector(f,x))

x,y = create_reverse_vector([[1, 2, 3], [1,3,6]])


# for multiple functions

def F1(array):
    x, y = create_reverse_vector(array)
    return 3*x + rop.cos(y)**2 + 1, x, y


def F2(array):
    x, y = create_reverse_vector(array)
    return rop.sin(x) + 2*rop.sin(y), x, y

array = [[1, 2, 3], [1,3,6]]
vec_functions = [F1(array), F2(array)]

for func in vec_functions:
    function, x, y = func
    print("The values of the function is ", function._val)
    print("The derivatives of the function with respect to values of variable x is", rad.partial_vector(func[0], x))
    print("The derivatives of the function with respect to values of variable y is", rad.partial_vector(func[0], y))





