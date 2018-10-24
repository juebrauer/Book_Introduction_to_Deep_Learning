import numpy as np

def f(x1,x2):

    w1 = x1
    w2 = x2
    dw1 = 1
    dw2 = 0
    result = 0
    if x1<x2:
        # f(x1,x2) = x1*x2 + sin(x1)
        # df/dx1 = x2 + cos(x1)
        w3 = w1 * w2
        dw3 = dw1 * w2 + w1 * dw2
        w4 = np.sin(w1)
        dw4 = np.cos(w1) * dw1
        w5 = w3 * w4
        dw5 = dw3 * w4 + w3 * dw4
        return w5, dw5
    else:
        # f(x1,x2) = pi + x1*x2 + 5*x1^2
        # df/dx1 = x2 + 10*x1
        w3 = np.pi
        dw3 = 0
        w4 = w1*w2
        dw4 = dw1*w2 + w1*dw2
        w5 = w3+w4
        dw5 = dw3 + dw4
        w7 = w5
        dw7 = dw5
        for i in range(0,5):
            w6 = w1**2
            dw6 = 2*w1
            w7 = w7 + w6
            dw7 = dw7 + dw6
        return w7, dw7

val, deriv = f(1,2)
print("f(1,2) = " + str(val))
print(2*np.sin(1))
print("dfdx1(1,2) = " + str(deriv))

print(np.cos(1))
print(np.sin(1))



