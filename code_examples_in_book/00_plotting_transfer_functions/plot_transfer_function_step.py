import numpy as np
import matplotlib.pyplot as plt

def f(x,T):
    if x < T:
        return 0
    else:
        return 1

x = np.arange(-4.0, 4.0, 0.01)

fig = plt.figure()
fig.suptitle('Step function (Heaviside)', fontsize=20)

plt.ylim(-1.2, 1.25)
plt.grid(True)
plt.xlabel('act', fontsize=14)
plt.ylabel('out = f(act)', fontsize=14)

Threshold = 1.2
y = [f(b, Threshold) for b in x]

plt.plot(x, y, 'r')

plt.annotate('Threshold T = ' + str(Threshold), xy=(Threshold, 0), xytext=(Threshold+0.5, -0.5),
            arrowprops=dict(facecolor='blue', shrink=0.01),
            )

fig.savefig('tf_step.png')

plt.show()
