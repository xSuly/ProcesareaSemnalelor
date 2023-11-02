#Laborator 4 - ex 1 - 2 nov
# Albei Liviu-Andrei === CTI 461

import numpy as np
import matplotlib.pyplot as plt
import time
from time import perf_counter

pi = np.pi
e = np.e

N = [128, 256, 512, 1024, 2048, 4096, 8192]
timpi_proprii = []
timpi_functie = []


#Fourier = np.ones((N, N), dtype=complex)
for counter in N:
    
    t = np.linspace(0, 2, counter)
    x = np.sin(2*pi*t)
    Fourier = np.ones((counter, counter), dtype=complex)
    start_time = time.time()
    for i in range(counter):
        for j in range(counter):
            Fourier[i][j] = e ** (2 * pi * 1j * i * j / counter)
    X = np.dot(Fourier, x)
    end_time = time.time()
    print("timp trecut pentru N = ", counter, " este ", end_time-start_time)
    timpi_proprii.append(end_time-start_time)
    start_time2 = perf_counter()
    Fourier2 = np.fft.fft(x)
    end_time2 = perf_counter()
    print("timp pentru functie cu N = ", counter, " este ", end_time2-start_time2)
    timpi_functie.append(end_time2-start_time2)

plt.figure(figsize=(12,6))
plt.plot(N, timpi_proprii, label="eu", marker='x')
plt.plot(N, timpi_functie, label="fct", marker='o')
#plt.xscale('log')
plt.yscale('log')
plt.xlabel('valori N')
plt.ylabel('Timp executie')
plt.title('Implementare proprie vs fft')
plt.show()

