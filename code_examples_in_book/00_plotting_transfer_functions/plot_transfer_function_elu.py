import numpy as np
import matplotlib.pyplot as plt

def f(x,a):
    if x > 0:
        return x
    else:
        return a*(np.exp(x)-1)

x = np.arange(-5.0, 5.0, 0.01)

fig = plt.figure()
fig.suptitle('Exponential Linear Unit (ELU)', fontsize=20)

plt.ylim(-3, 3)
plt.grid(True)
plt.xlabel('act', fontsize=14)
plt.ylabel('out = f(act)', fontsize=14)

a1=0.1
a2=0.3
a3=1.0

y1 = [f(b,a1) for b in x]
plt.plot(x, y1, 'r')

y2 = [f(b,a2) for b in x]
plt.plot(x, y2, 'g')

y3 = [f(b,a3) for b in x]
plt.plot(x, y3, 'b')

plt.annotate('a = '+str(a1), xy=(-4, -0.1), xytext=(-4, 1),
            arrowprops=dict(facecolor='red', shrink=0.01) )

plt.annotate('a = '+str(a2), xy=(-2, -0.25), xytext=(-2, 1),
            arrowprops=dict(facecolor='green', shrink=0.01) )

plt.annotate('a = '+str(a3), xy=(-1, -0.7), xytext=(1, -1.5),
            arrowprops=dict(facecolor='blue', shrink=0.01) )

fig.savefig('tf_elu.png')

plt.show()
