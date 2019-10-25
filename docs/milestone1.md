# Milestone 1

## Introduction

Automatic differentiation is a tool for calculating derivatives using machine accuracy. It has several advantages over traditional methods of derivative calculations such as symbolic and finite differentiation. Automatic differentiation is useful for calculating complex derivatives where errors are more likely with classical methods. For instance , with finite differentiation, h values that are too small will lead to accuracy errors though floating point roundoff error, while h values that are too large will start making vastly inaccurate approximations. 

Automatic differentiation is useful due to its practicality in real world applications that involve thousands of parameters in a complicated function, which would take a long runtime as well as strong possibility for error in calculating the derivatives individually. 

Our package allows users to calculate derivatives of complex functions, some with many parameters, allowing machine precision.

## Background

Essentially automatic differentiation works by  breaking down a complicated function and performing a sequence of elementary arithmetic such as addition, subtraction, multiplication, and division as well as elementary functions like exp, log, sin, etc. These operations are then repeated by the chain rule and the derivatives of these sequences are calculated. There are two ways that automatic differentiation can be implemented - forward mode and reverse mode. 

### 2.1 The Chain Rule

The chain rule makes up a fundamental component of auto differentiation. The basic idea is:   
For univariate function, $$ F(x) = f(g(x))$$
 $$F^{\prime} = (f(g))^{\prime} = f^{\prime}(g(x))g^{\prime}(x)$$
For multivariate function, $$F(x) = f(g(x),h(x))$$
$$ \frac{\partial F}{\partial x}=\frac{\partial f}{\partial g} \frac{\partial g}{\partial x}+\frac{\partial f}{\partial h} \frac{\partial h}{\partial x}$$
For more generalized cases, if F is a combination of more sub-functions,  $$F(x) = f(g_{1}(x), g_{2}(x), …, g_{m}(x))$$
$$\frac{\partial F}{\partial x}=\sum_{i=1}^{m}\frac{\partial F}{\partial g_{i}} \frac{\partial g_{i}}{\partial x}$$

### 2.2 Forward Mode

The forward mode automatic differentiation is accomplished by firstly splitting the function process into one-by-one steps, each including only one basic operation. Then from the first node, the value and derivative will be calculated based on the values and derivatives of forward nodes. An example of computational graph and table for forward mode AD is shown as follows:

\begin{align}
  f\left(x,y\right) =\sin\left(xy\right)
\end{align}
We will be evaluating the function at $f(1, 0)$

Evaluation trace:

| Trace   | Elementary Function      | Current Value           | Elementary Function Derivative       | $\nabla_{x}$ Value  | $\nabla_{y}$ Value  |
| :---: | :-----------------: | :-----------: | :----------------------------: | :-----------------:  | :-----------------: |
| $x_{1}$ | $x_{1}$                  | $1$        | $\dot{x}_{1}$                        | $1$ | $0$ |
| $x_{2}$ | $x_{2}$                  | $0$        | $\dot{x}_{2}$                        | $0$ | $1$ |
| $x_{3}$ | $x_{1}x_{2}$                  | $0$        | $\dot{x}_{2}$                        | $0$ | $1$ |

### 2.3 Reverse Mode

The reverse mode automatic differentiation has a process similar to the forward mode auto differentiation, but has another reverse process. It does not apply the chain rule and only partial derivatives to a node are stored. First, for the forward process, the partial derivatives are stored for each node. For the reverse process, it starts with the differentiation to the last node, and then activations in the forward process are deployed in the differentiation differentiation step by step. 


### 2.4 Forward Mode v.s. Reverse Mode

Two main aspects can be considered when choosing between Forward and Reverse mode auto differentiation.
* Memory Storage & Time of Computation

The forward mode needs memory storage for values and derivatives for each node, while the reverse mode only needs to store the activations of partial differentiation to each node. The forward mode do the computation at the same time as the variable evaluation, while the reverse mode do the calculation in the backward process.
* Input & Output Dimensionality

If the input dimension is much larger than output dimension, then reverse mode is more attractive. If the output dimension is much larger than the input dimension, the forward mode is much computational cheaper.





## How to use VorDiff

## Software Organization

## Implementation

