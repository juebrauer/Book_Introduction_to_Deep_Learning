# ---
# Python code to generate the logistic transfer function plot
#
# by Prof. Dr. Juergen Brauer, www.juergenbrauer.org
# ---

import numpy as np
import matplotlib.pyplot as plt

# Define the logistic transfer function
def f(x,gamma):
    return 1.0 / (1.0 + np.exp(-gamma*x))

# Prepare a vector of x-values
x = np.arange(-4.0, 4.0, 0.01)

# Set title of the plot
fig = plt.figure()
fig.suptitle('Logistic transfer function', fontsize=20)

# Set y-range to display, use a grid, set label for axes
plt.ylim(-0.25, 1.25)
plt.grid(True)
plt.xlabel('act', fontsize=14)
plt.ylabel('out = f(act)', fontsize=14)

# Plot 3 versions of the logistic transfer function
# using different values for gamma
plt.plot(x, [f(b,1.0)  for b in x], 'r')
plt.plot(x, [f(b,2.0)  for b in x], 'b')
plt.plot(x, [f(b,10.0) for b in x], 'g')

# Show arrows with annotation text which gamma value
# was used for which graph
plt.annotate('gamma = 1.0', xy=(1, 0.72), xytext=(2, 0.5),
            arrowprops=dict(facecolor='red', shrink=0.01),
            )

plt.annotate('gamma = 2.0', xy=(0.5, 0.72), xytext=(1.5, 0.3),
            arrowprops=dict(facecolor='blue', shrink=0.01),
            )

plt.annotate('gamma = 10.0', xy=(0.1, 0.8), xytext=(-3, 1.0),
            arrowprops=dict(facecolor='green', shrink=0.01),
            )

# Generate image file
fig.savefig('tf_logistic.png')

# Show the plot also on the screen
plt.show()
