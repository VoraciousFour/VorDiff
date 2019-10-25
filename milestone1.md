# Milestone 1

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
