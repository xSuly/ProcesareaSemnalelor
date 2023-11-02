#Laborator 3 - ex 3 - 26 oct
# Albei Liviu-Andrei === CTI 461

import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
e = np.e

n = 250
t = np.linspace(0, 2, n)
omega = [15, 55, 95]

X = 1.2 * np.sin(2*pi*omega[0]*t) + 0.2 * np.cos(2*pi*omega[1]*t) + 0.66 * np.sin(2*pi*omega[2]*t)

mat = np.zeros(n, dtype=complex)
for i in range(n):
    for j in range(n):
        mat[i] = mat[i] + X[j]*e**(-2*pi*1j*i*j/n) #din curs, la DFT

fig, axs = plt.subplots(2)
fig.suptitle('Modulul transformatei Fourier')
axs[0].plot(t, X)
axs[0].set_xlabel('t')
axs[0].set_ylabel('X')
axs[1].stem(np.arange(n), np.sqrt(mat.real**2 + mat.imag**2), markerfmt='x', linefmt='purple', basefmt=' ') #modulul transformatei
axs[1].set_xlabel('Frecventa')
axs[1].set_ylabel('Modul')

plt.savefig("Lab3/grafice/lab3_ex3.png")
plt.savefig("Lab3/grafice/lab3_ex3.pdf")
plt.show()