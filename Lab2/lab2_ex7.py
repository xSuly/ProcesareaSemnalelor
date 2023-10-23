#Laborator 2 - ex 7 - 19 oct
# Albei Liviu-Andrei === CTI 461

import numpy as np
import matplotlib.pyplot as plt

pi = np.pi

f0 = 1000 #Hz
t = np.linspace(0, 1, f0)

sinusoid_A = np.sin(2*pi*f0*t)

t_decimat = np.linspace(0, 1, int(f0/4))
#t_decimat = t_decimat[1::4]

sinusoid_B = np.sin(2*pi*t_decimat*f0)

fig, axs = plt.subplots(2)
fig.suptitle('Semnalele')
axs[0].plot(t, sinusoid_A)
axs[1].plot(t_decimat, sinusoid_B)

plt.show()
#E rezolvarea fara subpct B, nu inteleg exact cerinta, am incercat prin t_decimat[1::4] insa nu stiu cat de corect este