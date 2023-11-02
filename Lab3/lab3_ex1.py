#Laborator 3 - ex 1 - 26 oct
# Albei Liviu-Andrei === CTI 461

import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
e = np.e

N = 8
Fourier = np.ones((N, N), dtype=complex)
for i in range(N):
    for j in range(N):
        Fourier[i][j] = e ** (2 * pi * 1j * i * j / N)

fig, axs = plt.subplots(N, 2, figsize=(12, 6))
fig.suptitle('Partea imaginara si partea reala pentru matricea Fourier cu N = 8')

for i in range(N):
    axs[i][0].plot(np.arange(N), Fourier[i].real, 'purple')
    axs[i][1].plot(np.arange(N), Fourier[i].imag, 'orange')

plt.savefig("Lab3/grafice/lab3_ex1.png")
plt.savefig("Lab3/grafice/lab3_ex1.pdf")
plt.show()


conditie = np.allclose(np.dot(Fourier, Fourier.conjugate().T), N * np.identity(N))

if conditie:
    print("Mat Fourier este unitara")
else:
    print("Mat Fourier nu e unitara")