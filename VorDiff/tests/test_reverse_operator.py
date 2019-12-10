#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 18:15:20 2019

@author: diyixuan
"""

from VorDiff.reverse_operator import ReverseOperator as rop
from VorDiff.reverse_autodiff import ReverseAutoDiff as rad
import numpy as np
import math

x = rad.reverse_scalar(0.5)
c = 0.5 

def test_sin():
    
    # Scalar
    f = rop.sin(x)
    assert f._val == np.sin(x._val)
    assert f.compute_gradient() == np.cos(x._val)*x.compute_gradient()

    #Constant
    assert rop.sin(c) == np.sin(c)
    

def test_cos():
    
    # Scalar
    f = rop.cos(x)
    assert f._val == np.cos(x._val)
    assert f.compute_gradient() == -np.sin(x._val)*x.compute_gradient()

    #Constant
    assert rop.cos(c) == np.cos(c)

    
def test_tan():
    
    # Scalar
    f = rop.tan(x)
    assert f._val == np.tan(x._val)
    assert f.compute_gradient() == x.compute_gradient()/np.cos(x._val)**2

    #Constant
    assert rop.tan(c) == np.tan(c)


    
def test_arcsin():
    
    # Scalar
    f = rop.arcsin(x)
    assert f._val == np.arcsin(x._val)
    assert f.compute_gradient() == 1/(x.compute_gradient()*(1-x._val**2)**.5)
    
    #Constant
    assert rop.arcsin(c) == np.arcsin(c)


    
def test_arccos():
    
    # Scalar
    f = rop.arccos(x)
    assert f._val == np.arccos(x._val)
    assert f.compute_gradient() == -x.compute_gradient()/(1-x._val**2)**.5

    #Constant
    assert rop.arccos(c) == np.arccos(c)


    
def test_arctan():
    
    # Scalar
    f = rop.arctan(x)
    assert f._val == np.arctan(x._val)
    assert f.compute_gradient() == x.compute_gradient()/(1+x._val**2)

    #Constant
    assert rop.arctan(c) == np.arctan(c)
    


def test_log():

    # Scalar
    f = rop.log(x)
    assert f._val == np.log(x._val)
    assert f.compute_gradient() == x.compute_gradient()/x._val

    #Constant
    assert rop.log(c) == np.log(c)



def test_sinh():
    
    # Scalar
    f = rop.sinh(x)
    assert f._val == np.sinh(x._val)
    assert f.compute_gradient() == x.compute_gradient()*np.cosh(x._val)
    #Constant
    assert rop.sinh(c) == np.sinh(c)



def test_cosh():

    # Scalar
    f = rop.cosh(x)
    assert f._val == np.cosh(x._val)
    assert f.compute_gradient() == x.compute_gradient()*np.sinh(x._val)

    #Constant
    assert rop.cosh(c) == np.cosh(c)



def test_tanh():

    # Scalar
    f = rop.tanh(x)
    assert f._val == np.tanh(x._val)
    assert f.compute_gradient() == x.compute_gradient()*(1-np.tanh(x._val)**2)
    
    #Constant
    assert rop.tanh(c) == np.tanh(c)






def test_sqrt():

    #scalar
    f = rop.sqrt(x)
    assert f._val == x._val**(0.5)
    assert round(f.compute_gradient(),5) == round(x.compute_gradient()*(x._val**(-1/2)/2),5)


    #constant
    assert rop.sqrt(c) == c**0.5






test_sin()
test_cos()
test_tan()
test_arcsin()
test_arccos()
test_arctan()
test_sinh()
test_cosh()
test_tanh()
test_log()
test_sqrt()


