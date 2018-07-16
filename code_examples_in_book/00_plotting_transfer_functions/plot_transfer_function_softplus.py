import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.log(1+np.exp(x))

x = np.arange(-7.0, 7.0, 0.01)

fig = plt.figure()
fig.suptitle('Softplus', fontsize=20)

plt.ylim(-7.25, 7.25)
plt.grid(True)
plt.xlabel('act', fontsize=14)
plt.ylabel('out = f(act)', fontsize=14)

y = [f(b) for b in x]

plt.plot(x, y, 'r')

fig.savefig('tf_softplus.png')

plt.show()
