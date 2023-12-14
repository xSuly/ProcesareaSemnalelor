#Laborator 9 - ex 2 - 14 decembrie
# Albei Liviu-Andrei === CTI 461

import numpy as np
import matplotlib.pyplot as plt

pi = np.pi

N = 5
x = np.arange(N)

trend = 5*x**2 - 2*x + 10
sezon = np.sin(3*pi*2*x) + np.sin(7*pi*2*x)
variatii_mici = np.random.normal(0, 1, N)

serie_timp = trend + sezon + variatii_mici

alfa = 0.25
s = np.zeros(len(serie_timp))

for t in range(len(serie_timp)):
    print(t)

for t in range(len(serie_timp)):
    suma = 0
    for j in range(1, t+1):
        suma = suma + ((1-alfa)**(t-j)) * serie_timp[j]
    #suma = suma * 1e-2
    s[t] = alfa * suma + ((1-alfa)**t)*serie_timp[0]


print(serie_timp[0])
print(s[1])
#print(s)
plt.figure(figsize=(12, 6))
plt.plot(x, serie_timp, label='seria originala', color = 'red')
plt.plot(x, s, label='medierea', color = 'blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Exercitiu2')
plt.legend(loc='lower right')
plt.show()



