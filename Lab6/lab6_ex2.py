#Laborator 6 - ex 2 - 16 nov
# Albei Liviu-Andrei === CTI 461

import numpy as np
import matplotlib.pyplot as plt

N = 2
valoare = 5

p = np.random.randint(-valoare, valoare, N + 1)
q = np.random.randint(-valoare, valoare, N + 1)

#print(p)
#print(q)
r = np.convolve(p, q)
r2 = np.real(np.fft.ifft(np.fft.fft(p, n=2*N+1) * np.fft.fft(q, n=2*N+1))) #folosim 2*N+1 pentru ca dupa ce inmultim 2 polinoame atatia coeficienti vom avea in noul polinom si trebuie sa umplem cu 0 pana la 2*N+1 sa nu se piarda valori

print(r)

print('---------------')

print(np.real(r2))


