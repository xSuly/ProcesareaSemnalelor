#Laborator 6 - ex 1 - 16 nov
# Albei Liviu-Andrei === CTI 461

import numpy as np
import matplotlib.pyplot as plt

N = 100
vector = np.random.rand(N)

iteratii = [vector]

for i in range(3):
    vector = np.convolve(vector, vector)
    iteratii.append(vector)
#print(vector)

fig, axs = plt.subplots(4)
fig.suptitle('Exercitiu1 lab6')
axs[0].plot(iteratii[0])
axs[0].set_title('0 iteratii')
axs[1].plot(iteratii[1])
axs[1].set_title('o iteratie')
axs[2].plot(iteratii[2])
axs[2].set_title('doua iteratii')
axs[3].plot(iteratii[3])
axs[3].set_title('trei iteratii')
plt.tight_layout
plt.show()

