#Laborator 6 - ex 3 - 16 nov
# Albei Liviu-Andrei === CTI 461

import numpy as np
import matplotlib.pyplot as plt

pi = np.pi


fs = 100
Nw = 200
t = np.linspace(0, 1, Nw)
x = np.sin(2*pi*fs*t) #A = 1 deci sinusoida unitara
#phi = 0

def fereastra_dreptunghiulara(X):
    w = np.ones(len(X))
    return w

def fereastra_Hanning(X):
    w = 0.5 * (1 - np.cos(2*pi*X/len(X)))
    return w

plt.figure(figsize=(12, 6))
plt.plot(t, fereastra_dreptunghiulara(x)*x)
plt.plot(t, fereastra_Hanning(x)*x)
plt.title('Exercitiu3')
plt.legend(loc='lower right')
plt.show()

