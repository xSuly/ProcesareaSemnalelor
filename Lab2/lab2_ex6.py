#Laborator 2 - ex 6 - 19 oct
# Albei Liviu-Andrei === CTI 461

import numpy as np
import matplotlib.pyplot as plt

pi = np.pi

t = np.linspace (0, 2, 100)
fs = 685 #Hz, frecventa de esantionare

fs_a = fs/2
fs_b = fs/4
fs_c = 0

#Avem amplitudine unitara (1) si faza nula (0)

sinusoid_1 = np.sin(2*pi*fs_a*t)
sinusoid_2 = np.sin(2*pi*fs_b*t)
sinusoid_3 = np.sin(2*pi*fs_c*t)

fig, axs = plt.subplots(3)
fig.suptitle('Semnalele')
axs[0].plot(t, sinusoid_1)
axs[1].plot(t,sinusoid_2)
axs[2].plot(t,sinusoid_3)

plt.show()

#Notita: observam ca atunci cand frecventa scade, numarul deviatiilor (coborari~urcari) ale semnalului alterneaza mai mult decat atunci cand frecventa
#este mai mare, ajungand pana in punctul de linie dreapta, in cazul 0