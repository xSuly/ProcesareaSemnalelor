#Laborator 2 - ex 7 - 19 oct
# Albei Liviu-Andrei === CTI 461

import numpy as np
import matplotlib.pyplot as plt

pi = np.pi

f0 = 1000 #Hz
t = np.linspace(0, 1, f0)

sinusoid_A = np.sin(2*pi*f0*t)

sinusoid_A_decimat = sinusoid_A[::4]
f_decimat = int(f0/4)
t_decimat = np.linspace(0, 1, f_decimat)

#sinusoid_B = np.sin(2*pi*t_decimat*f0)

sinusoid_A_2start = sinusoid_A[1::4]

fig, axs = plt.subplots(3)
fig.suptitle('Semnalele')
axs[0].stem(t, sinusoid_A)
axs[1].stem(t_decimat, sinusoid_A_decimat)
axs[2].stem(t_decimat, sinusoid_A_2start)

plt.show()
#E rezolvarea fara subpct B, nu inteleg exact cerinta, am incercat prin t_decimat[1::4] insa nu stiu cat de corect este
#Corectie dupa intrebarile de la inceputul LAB4 -> finish a, adaugat subpct b