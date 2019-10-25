# Milestone1

## Introduction

## Background

## How to use VorDiff

The two main objects you will interact with are imported below:
```py
from VorDiff import autodiff as ad
from VorDiff import operator as op
```
In short, the user will first instantiate a variable (scalar or vector) as an `autodiff` object, and then feed those variables to operators in the `operator` object. The user will be able to retrieve the resulting values and derivatives after doing so.

The user may instantiate an `autodiff` object by the following:
```py
x = ad.create_scalar(val = 4)
v = ad.create_vector(vals = [4,2]
```
Then, the `operator` class can be used to define functions. Simple operations (addition, subtraction, multiplication, and division) may be used normally. More complex functions (e.g. sqrt, sin, cos) must use the operations defined in the `operator` class.
```py
fx = 4*op.sqrt(x) + x
fv = 4*op.sqrt(v) + v
```
The user may retrieve the values and first derivatives from the objects defined above by using the `val` and `der` attributes of the `autodiff` object.
```py
print(fx.val, fx.der)
print(fv.val, fv.der)
```
For vectors, `val` and `der` will be tuples that contain the values and partial derivatives, respectively, at each element of the vector.

## Software Organization

### Directory Structure
The package's directory will be structured as follows:
```
VorDiff/
	__init__.py
	operators/
	    __init__.py
	    operator.py
	nodes/
	    __init__.py
	    node.py
	    scalar.py
	    vector.py
	tests/
	    __init__.py
	    test_operators/
		    test_operator.py
	    test_nodes/
		    test_node.py
		    test_scaler.py
		    test_vector.py
		test_autodiff.py
	docs/
		...
	examples/
	    __init__.py
	    ...
	autodiff.py 
    README.md
    setup.py
    ...
```
### Modules
-   VorDiff: The VorDiff module contains the operator class to be directly used by users to evaluate functions and calculate their derivatives, a superclass node and subclasses scalar and vector to be used in autodiff class, and an autodiff class to perform automatic differentiation. This is the core of the package.
    
-   Test_Vordiff: The Test_Vordiff module contains the test suite for this project. TravisCI and CodeCov are used to test our operator classes, node classes, and auto-differentiator.
    
-   Examples: The Examples module contains python files demonstrating how to perform automatic differentiation with the implemented functions.
    
### Testing
In this project we will use TravisCI to perform continuous integration testing and CodeCov to check the code coverage of our test suite. The status us TravisCI and CodeCov can be found in README.md, in the top level of our package. Since the test suite is included in the project distribution, users can also install the project package and use pytest and pytest-cov to check the test results locally.

### Distribution:
Our open-source VorDiff package will be uploaded to PyPI by using twine because it uses a verified connection for secure authentication to PyPI over HTTPS. Users will be able to install our project package by using the convential `pip install VorDiff`.

## Implementation

### Node
The `Node()` class represents a single node in our computational graph. It outlines the operations necessary for the `scalar` and `vector` objects to implement. These two types of variables are separated into their own classes to reduce complexity and improve organization.
```py
class Node():

    def __init__(self, val: float, der: float):
        self.val = val
        self.der = der

   def __add__(self, other):
       raise NotImplementedError

   def __radd__(self, other):
       raise NotImplementedError

   def __mul__(self, other):
       raise NotImplementedError

   def __rmul__(self, other):
       raise NotImplementedError
   ...
   ```

### Operator
The operator class contains all mathematical operations that users can call to build their functions. Each function returns a `Node` object.
```py
from ... import Node

class Operator():

    def __init__(self):
        pass

    def sqrt(self, x):
        pass

    def sin(self, x):
        pass
```

The elementary function objects are analytically differentiable and some are defined in the numpy package. We will include the following elementary functions:

- Trigonometric functions: sin, cos, tan
- Inverse Trigonometric functions: arcsin, arccos, arctan
- Hyperbolic functions: sinh, cosh, tanh, arcsinh, arccosh, arctanh
- Exponents: exp, exp2, expm1
- Logarithms: log, log2, log10, log1p

The derivatives of these elementary function objects and the self-defined elementary function objects (such as power and sqrt) can be evaluated with the chain rule.

### AutoDiff
The `AutoDiff` class will allow the user to easily create variables and build auto-differentiable functions, without having to interface with the `Node` class. It will make use of the auto-differentiator much more intuitive for the user.
```py
from Nodes import Scalar
from Nodes import Vector

class AutoDiff():
    def __init__(self):
        pass

    def create_scalar(self, val: float = 0):
        pass

    def create_vector(self, vals):
        pass
```

### External dependencies:
We will use numpy package for mathematical computation (such as sin, log, exp), and pytest and pytest-cov for the test suite.
