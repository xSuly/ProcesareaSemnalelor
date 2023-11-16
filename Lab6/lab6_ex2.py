#Laborator 6 - ex 2 - 16 nov
# Albei Liviu-Andrei === CTI 461

import numpy as np
import matplotlib.pyplot as plt

N = 2
valoare = 5

p = np.random.randint(-valoare, valoare, N + 1)
q = np.random.randint(-valoare, valoare, N + 1)

print(p)
print(q)
r = np.convolve(p, q)
r2 = np.fft.fft(np.fft.fft(p) * np.fft.fft(q))



print(r)

print('---------------')

print(np.real(r2))


