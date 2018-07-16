import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.tanh(x)

x = np.arange(-5.0, 5.0, 0.01)

fig = plt.figure()
fig.suptitle('Hyperbolic Tangent', fontsize=20)

plt.ylim(-1.25, 1.25)
plt.grid(True)
plt.xlabel('act', fontsize=14)
plt.ylabel('out = f(act)', fontsize=14)

y = [f(b) for b in x]

plt.plot(x, y, 'r')

fig.savefig('tf_tanh.png')

plt.show()
