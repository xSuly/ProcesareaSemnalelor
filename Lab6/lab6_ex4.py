#Laborator 6 - ex 4 - 16 nov
# Albei Liviu-Andrei === CTI 461

import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
e = np.e

x = np.genfromtxt('Lab5/Train.csv', delimiter=',', skip_header=1)

#subpct A, impartire in 3 zile
start_index = np.where(x[:, 0] == 1500)[0][0]

valori_trei_zile = x[start_index:start_index + 90] #90 adica 3 zile * 24 de ore - fiecare ora are o numaratoare separata

numara_coloane = valori_trei_zile[:, 2]

X = np.fft.fft(numara_coloane)
t1 = np.arange(0, len(numara_coloane))

#Subpct b
#w = 5
#netezire = np.convolve(X, np.ones(w), 'valid')
#t = np.arange(0, len(netezire))

plt.figure(figsize=(12, 6))
plt.plot(t1, abs(X)) 
plt.xlabel('Esantionare')
plt.ylabel('Nr masini')
#plt.yscale('log')
plt.title('Exercitiu4 Sub_pctA')
plt.grid()
plt.show()

w_values = [5, 9, 13, 17]

plt.figure(figsize=(12, 6))

for w in w_values:
    netezire = np.convolve(X, np.ones(w), 'valid') / w
    t2 = np.arange(0, len(netezire))
    plt.plot(t2, abs(netezire), label=f'Fereastra {w}')

plt.xlabel('Esantionare')
plt.ylabel('Nr masini')
plt.title('Exercitiu4 Sub_pctB - Filtru Medie Alunecatoare')
plt.legend()
plt.grid()
plt.show()