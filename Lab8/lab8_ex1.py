#Laborator 8 - ex 1 - 7 decembrie
# Albei Liviu-Andrei === CTI 461

import numpy as np
import matplotlib.pyplot as plt

pi = np.pi

N = 1000
x = np.arange(N)

trend = 5*x**2 - 2*x + 10
sezon = np.sin(3*pi*2*x) + np.sin(7*pi*2*x)
variatii_mici = np.random.normal(0, 1, N)

serie_timp = trend + sezon + variatii_mici

fig, axs = plt.subplots(4)
fig.suptitle('Exercitiu1 lab8')
axs[0].plot(x, serie_timp, 'red')
axs[0].set_title('serie timp')
axs[1].plot(x, trend, 'blue')
axs[1].set_title('trend')
axs[2].plot(x, sezon, 'magenta')
axs[2].set_title('sezon')
axs[3].plot(x, variatii_mici, 'green')
axs[3].set_title('variatii mici')
plt.tight_layout()
plt.show()

#subpct B

autocorelatie = np.correlate(serie_timp, serie_timp, mode = "full")


plt.figure(2)

plt.plot(x, autocorelatie)

plt.title('Vectorul de autocorelație al seriei de timp')
plt.show()

