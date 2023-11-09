#Laborator 5 - ex 1 - 9 nov
# Albei Liviu-Andrei === CTI 461

import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
e = np.e

x = np.genfromtxt('Lab5/Train.csv', delimiter=',', skip_header=1)

numara_coloane = x[:, 2]
coloane_filtrare = numara_coloane[numara_coloane < 5]

X = np.fft.fft(coloane_filtrare)

t = np.arange(0, len(coloane_filtrare))
print(np.arange(0, len(coloane_filtrare)))

plt.figure(figsize=(12, 6))
plt.plot(t, abs(X)) 
plt.xlabel('Esantionare')
plt.ylabel('Nr masini')
plt.title('Exercitiu1 Sub_pctD')
plt.show()

