import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return max(0,x)

x = np.arange(-7.0, 7.0, 0.01)

fig = plt.figure()
fig.suptitle('Rectified Linear Unit (ReLU)', fontsize=20)

plt.ylim(-7.25, 7.25)
plt.grid(True)
plt.xlabel('act', fontsize=14)
plt.ylabel('out = f(act)', fontsize=14)

y = [f(b) for b in x]

plt.plot(x, y, 'r')

fig.savefig('tf_relu.png')

plt.show()
