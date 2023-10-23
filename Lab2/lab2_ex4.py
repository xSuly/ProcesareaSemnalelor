#Laborator 2 - ex 4 - 19 oct
# Albei Liviu-Andrei === CTI 461

import numpy as np
import matplotlib.pyplot as plt

pi = np.pi

#ambele semnale au fost folosite si in lab1, ex2, insa pentru primul avem 650 la esantionare pentru a fi egal
#semnal sawtooth
f0_a = 240 #Hz
t_a = np.linspace(0, 2, 650) 
sawtooth_a = 2*(t_a*f0_a - np.floor(t_a*f0_a + 1/2))

#semnal square
f0_b = 300 #Hz
t_b = np.linspace(0, 2, 650) 
square_b = np.sign(np.sin(2*pi*t_b*f0_b))

suma_semnale = sawtooth_a + square_b

fig, axs = plt.subplots(3)
fig.suptitle('Semnalele')
axs[0].plot(t_a,sawtooth_a)
axs[1].plot(t_b,square_b)
axs[2].plot(t_a,suma_semnale)

plt.show()
