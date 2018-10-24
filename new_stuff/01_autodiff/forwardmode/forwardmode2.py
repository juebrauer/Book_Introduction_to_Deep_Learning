import numpy as np

def f(x1,x2):

    w1 = x1
    dw1 = 1
    w2 = x2
    dw2 = 0
    w3 = w1*w2
    dw3 = dw1 * w2 + w1 * dw2
    w4 = np.sin(w1)
    dw4 = np.cos(w1) * dw1
    w5 = w3 + w4
    dw5 = dw3 + dw4
    return w5, dw5


val, deriv = f(1,2)
print("df/dx1 (1,2)=" + str(deriv))
print("df/dx1 (1,2)=" + str(2 + np.cos(1))) # df/dx1 = x2 + cos(x1)
