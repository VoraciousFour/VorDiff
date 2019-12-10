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
assert rad.partial_scalar(f) == -0.4693956404725932


# for vector
x,y = create_reverse_vector([[1, 2, 3], [1,3,6]])

a = x + 1
assert (rad.partial_vector(a,x) == [1.0, 1.0, 1.0]).all()

x,y = create_reverse_vector([[1, 2, 3], [1,3,6]])

h = rop.sin(x)
partial_derivative = rad.partial_vector(h,x)
answer = [ 0.54030231, -0.41614684, -0.9899925 ]
for i in range(len(partial_derivative)):
    partial_derivative[i] = round(partial_derivative[i], 3)
    answer[i] = round(answer[i],3)
assert (partial_derivative == answer).all()
