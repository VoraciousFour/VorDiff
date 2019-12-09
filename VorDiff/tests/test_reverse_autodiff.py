#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 18:48:23 2019

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





