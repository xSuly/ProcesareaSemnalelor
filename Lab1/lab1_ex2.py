#Laborator 1 - ex 2 - 16oct
# Albei Liviu-Andrei === CTI 461

import numpy as np
import matplotlib.pyplot as plt


pi = np.pi
f0 = 400
t0 = 4 #timpul

t = np.linspace(0, 4, 1600) #esantionarea, cele 1600 de esantioane din ipoteza

#Subpct a
sinusoid_A = np.sin(2*pi*f0*t)

plt.figure(figsize=(12, 6))
plt.plot(t, sinusoid_A)
plt.xlabel('Timp')
plt.ylabel('Amplitudine')
plt.title('Exercitiu2 subpct A')
plt.legend(loc='lower right')
plt.show()

#Subpct b
f0_B = 800 #Hz
t_B = np.linspace(0, 3, 1600) #1600 de esantioane in 3 secunde

sinusoid_B = np.sin(2*pi*f0_B*t_B)
plt.figure(figsize=(12, 6))
plt.plot(t_B, sinusoid_B, color = 'yellow')
plt.xlabel('Timp')
plt.ylabel('Amplitudine')
plt.title('Exercitiu2 subpct B')
plt.legend(loc='lower right')
plt.show()

#Subpct c

#Semnalul sawtooth/ dinti de fierastrau are formula x(t) = 2(t - [t + 1/2]) -> [] reprezinta partea intreaga, rezolvam cu np.floor

f0_C = 240 #Hz
t_C = np.linspace(0, 1, 500) #500 esantioane intr-o secunda

sawtooth_C = 2*(t_C*f0_C - np.floor(t_C*f0_C + 1/2)) #inmultim la t_C cu frecventa respectiva
plt.figure(figsize=(12, 6))
plt.plot(t_C, sawtooth_C, color = 'green')
plt.xlabel('Timp')
plt.ylabel('Amplitudine')
plt.title('Exercitiu2 subpct C')
plt.legend(loc='lower right')
plt.show()

#Subpct d

#Semnalul square -> formula x(t) = sign(sin(2*pi*t*frecv))

f0_D = 300 #Hz
t_D = np.linspace(0, 2, 650) #650 esantioane in 2 secunde

square_D = np.sign(np.sin(2*pi*t_D*f0_D))
plt.figure(figsize=(12, 6))
plt.plot(t_D, square_D, color = 'red')
plt.xlabel('Timp')
plt.ylabel('Amplitudine')
plt.title('Exercitiu2 subpct D')
plt.legend(loc='lower right')
plt.show()


#Subpct E

sinusoid_E = np.random.rand(128, 128) #semnal 2D aleatoriu
plt.figure(figsize=(12, 6))
plt.imshow(sinusoid_E, cmap='PuBu_r') #nu stiu ce face imshow, asa scrie in cerinta sa folosesc
plt.xlabel('Timp')
plt.ylabel('Amplitudine')
plt.title('Exercitiu2 subpct E')
plt.legend(loc='lower right')
plt.show()

#Subpct F

#sinusoid_F = np.zeros(128, 128)
sinusoid_F = [[(i*1.5 + j - 2) * (j*10.5 - i + 2) for j in range (128)] for i in range (128)]
plt.figure(figsize=(12, 6))
plt.imshow(sinusoid_F, cmap='gist_rainbow')
plt.xlabel('Timp')
plt.ylabel('Amplitudine')
plt.title('Exercitiu2 subpct F')
plt.legend(loc='lower right')
plt.show()


