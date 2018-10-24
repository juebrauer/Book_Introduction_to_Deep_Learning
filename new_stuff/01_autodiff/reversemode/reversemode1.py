import numpy as np

def f(x1,x2):

    # forward step
    w1 = x1
    w2 = x2
    w3 = w1*w2
    w4 = np.sin(w1)
    w5 = w3 * w4
    # so f(x1,x2) = x1*x2*sin(x1)

    # backward step
    _w5 = 1
    _w3 = 1 * w4
    _w4 = 1 * w3
    _w1a = w4 * w2
    _w1b = w3 * np.cos(w1)
    _w1 = _w1a+_w1b
    _w2 = w4 * w1

    return w5, _w1, _w2

val, deriv1, deriv2 = f(1,2)
print("f(1,2)="+str(val))
print("df/dx1(1,2)="+str(deriv1))
print("df/dx2(1,2)="+str(deriv2))
