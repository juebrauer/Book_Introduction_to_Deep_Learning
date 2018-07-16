import numpy as np
import matplotlib.pyplot as plt

def f(x,T1,T2):
    if x < T1:
        return 0
    elif T1 <= x and x <= T2:
        return (x-T1)*(T2-T1)
    else:
        return 1

x = np.arange(-2.0, 6.0, 0.01)

fig = plt.figure()
fig.suptitle('Ramp function', fontsize=20)

plt.ylim(-1.2, 1.25)
plt.grid(True)
plt.xlabel('act', fontsize=14)
plt.ylabel('out = f(act)', fontsize=14)

Threshold1 = 1.2
Threshold2 = 2.2
y = [f(b, Threshold1, Threshold2) for b in x]

plt.plot(x, y, 'r')

plt.annotate('Threshold T1 = ' + str(Threshold1), xy=(Threshold1, 0), xytext=(Threshold1-0.5, -0.75),
            arrowprops=dict(facecolor='blue', shrink=0.01))

plt.annotate('Threshold T2 = ' + str(Threshold2), xy=(Threshold2, 0), xytext=(Threshold2+1.5, -0.5),
            arrowprops=dict(facecolor='blue', shrink=0.01))

x1, y1 = [Threshold1, Threshold1], [0, 0]
plt.plot(x1, y1, marker = 'o', color='gray')

x1, y1 = [Threshold2, Threshold2], [1, 0]
plt.plot(x1, y1, '--', marker = 'o', color='gray')


fig.savefig('tf_ramp.png')

plt.show()
