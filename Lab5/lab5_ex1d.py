#Laborator 5 - ex 1 - 9 nov
# Albei Liviu-Andrei === CTI 461

import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
e = np.e

x = np.genfromtxt('Lab5/Train.csv', delimiter=',', skip_header=1)
#X = np.fft.fft(x)

numara_coloane = x[:, 2]

X = np.fft.fft(numara_coloane)

t = np.arange(0, len(numara_coloane))
print(np.arange(0, len(numara_coloane)))

jumatate = len(t) // 2

plt.figure(figsize=(12, 6))
#plt.plot(t, x[:, 2]) # pentru afisarea graficului din laborator
plt.plot(t[:jumatate], abs(X[:jumatate])) #pentru afisare grafica modul transformata
plt.xlabel('Esantionare')
plt.ylabel('Nr masini')
plt.title('Exercitiu1 Sub_pctD')
plt.show()

