import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x

x = np.arange(-4.0, 4.0, 0.1)

fig = plt.figure()
fig.suptitle('Identity', fontsize=20)

plt.ylim(-4.0, 4.0)
plt.grid(True)
plt.xlabel('act', fontsize=14)
plt.ylabel('out = f(act)', fontsize=14)

plt.plot(x, f(x), 'r')

fig.savefig('tf_identity.png')

plt.show()
