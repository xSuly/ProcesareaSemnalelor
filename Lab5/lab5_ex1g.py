#Laborator 5 - ex 1 - 9 nov
# Albei Liviu-Andrei === CTI 461

import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
e = np.e

x = np.genfromtxt('Lab5/Train.csv', delimiter=',', skip_header=1)

id_coloana = 0

start_index = np.where(x[:, id_coloana] == 1500)[0][0]

valori_o_luna = x[start_index:start_index + 720] #720 adica 30 de zile * 24 de ore - fiecare ora are o numaratoare separata

numara_coloane = valori_o_luna[:, 2]

X = np.fft.fft(numara_coloane)

t = np.arange(0, len(numara_coloane))

plt.figure(figsize=(12, 6))
plt.plot(t, abs(X)) 
plt.xlabel('Esantionare')
plt.ylabel('Nr masini')
plt.yscale('log')
plt.title('Exercitiu1 Sub_pctG')
plt.show()

