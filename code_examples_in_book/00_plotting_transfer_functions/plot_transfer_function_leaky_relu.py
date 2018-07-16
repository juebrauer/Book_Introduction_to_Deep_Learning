import numpy as np
import matplotlib.pyplot as plt

def f(x):
    if x > 0:
        return x
    else:
        return 0.03*x

x = np.arange(-2.0, 2.0, 0.01)

fig = plt.figure()
fig.suptitle('Leaky ReLU', fontsize=20)

plt.ylim(-0.5, 2.25)
plt.grid(True)
plt.xlabel('act', fontsize=14)
plt.ylabel('out = f(act)', fontsize=14)

y = [f(b) for b in x]

plt.plot(x, y, 'r')

fig.savefig('tf_leaky_relu.png')

plt.show()
