import numpy as np

p1 = 1.2345
p2 = 9.8765


def f(w1,w2):

    # forward step
    w3 = p1*w1
    w4 = w3 + w2         #  p1*w1+w2
    w5 = w4**2           # (p1*w1+w2)^2
    w6 = p2 * w4**2      #  p2*(p1*w1+w2)^2
    w7 = np.sin(w6)      #  sin(p2*(p1*w1+w2)^2))
    w8 = w2*w7           #  w2*sin(p2*(p1*w1+w2)^2))
    # so f(x1,x2) = w8 = w2*sin(p2*(p1*w1+w2)^2))

    # backward step
    _w8 = 1.0
    _w7 = w2
    _w6 = _w7 * np.cos(w6)
    _w5 = _w6 * p2
    _w4 = _w5 * 2.0 * w4
    _w3 = _w4 * 1.0
    _w1 = _w3 * p1
    _w2a = _w4 * 1.0
    _w2b = _w8 * w7
    _w2 = _w2a+_w2b

    # return the function value w8
    # and the two derivatives
    # df/dw1=_w1 and
    # df/dw2=_w2
    return w8,_w1,_w2


def numdiff_dfdw1(w1,w2):
    h = 0.000001
    f1,_,_ = f(w1+h,w2)
    f2,_,_ = f(w1,w2)
    return (f1 - f2 ) / h

def numdiff_dfdw2(w1,w2):
    h = 0.000001
    f1, _, _ = f(w1, w2+h)
    f2, _, _ = f(w1, w2)
    return (f1 - f2) / h

def manualdiff_dfdw1(w1,w2):
    deriv = w2*np.cos(p2*(p1*w1+w2)**2)*p2*2*(p1*w1+w2)*p1
    return deriv

def manualdiff_dfdw2(w1,w2):
    deriv = 1*np.sin(p2*(p1*w1+w2)**2) + w2*np.cos(p2*(p1*w1+w2)**2)*p2*2*(p1*w1+w2)*1
    return deriv


def main():
    y,dfdw1,dfdw2 = f(3, 4)
    print("f(3,4) = " + str(y))

    print("Numerical differentiation --> dfdw1(3,4) = " + str(numdiff_dfdw1(3, 4)))
    print("Numerical differentiation --> dfdw2(3,4) = " + str(numdiff_dfdw2(3, 4)))

    print("Manual differentiation --> dfdw1(3,4) = " + str(manualdiff_dfdw1(3, 4)))
    print("Manual differentiation --> dfdw2(3,4) = " + str(manualdiff_dfdw2(3, 4)))

    print("Reverse-Mode Autodiff --> dfdw1(3,4) = " + str(dfdw1))
    print("Reverse-Mode Autodiff --> dfdw2(3,4) = " + str(dfdw2))

main()







