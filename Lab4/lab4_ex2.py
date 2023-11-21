#Laborator 4 - ex 2 - 2 nov
# Albei Liviu-Andrei === CTI 461

import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
e = np.e

t = np.linspace(0, 2, 666)
t2 = np.linspace(0, 2, 5) #5puncte

f0 = 14

fig, axs = plt.subplots(4)
axs[0].plot(t, np.sin(2*pi*f0*t), c = 'cyan')

axs[1].plot(t, np.sin(2*pi*f0*t), c = 'cyan')
axs[1].scatter(t2, np.sin(2*pi*f0*t2), c = 'yellow')

axs[2].plot(t, np.sin(2*pi*(f0-11)*t), c = 'purple')
axs[2].scatter(t2, np.sin(2*pi*f0*t2), c = 'yellow')

axs[3].plot(t, np.sin(2*pi*(f0-8)*t), c = 'green')
axs[3].scatter(t2, np.sin(2*pi*f0*t2), c = 'yellow')

plt.tight_layout()
plt.show()