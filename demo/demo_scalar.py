import sys
sys.path.append("..")
from VorDiff.nodes.scalar import Scalar
from VorDiff.operator import Operator as op
from VorDiff.autodiff import AutoDiff as ad

x1 = ad.scalar(2)
x2 = 2*x1+1
x3 = x2**2
x4 = x3/x1
x5 = op.sin(x4)
x6 = op.exp(x5)

print("The value and derivative of x1 is", x1.get())
print("The value and derivative of x2 with respect to x1 is", x2.get())
print("The value and derivative of x3 with respect to x1 is", x3.get())
print("The value and derivative of x4 with respect to x1 is", x4.get())
print("The value and derivative of x5 with respect to x1 is", x5.get())
print("The value and derivative of x6 with respect to x1 is", x6.get())
