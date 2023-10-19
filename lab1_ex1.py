#Laborator 1 - ex 1 - 16oct
# Albei Liviu-Andrei === CTI 461

import numpy as np
import matplotlib.pyplot as plt



#Esantionare
t = np.arange(0, 0.03, 0.0005)

pi = np.pi

#Functiile
xt = np.cos(520*pi*t + pi/3)
yt = np.cos(280*pi*t - pi/3)
zt = np.cos(120*pi*t + pi/3)

"""
#Subpct a
plt.figure(figsize=(12, 6))
plt.plot(t, xt, label='x(t)= 520*pi*t + pi/3')
plt.plot(t, yt, label='y(t)= 280*pi*t - pi/3')
plt.plot(t, zt, label='z(t)= 120*pi*t + pi/3')
plt.xlabel('Timp')
plt.ylabel('b')
plt.title('Exercitiu1 Sub_pctA')
plt.legend(loc='lower right')
plt.show()
"""
"""
#Subpct b
fig, axs = plt.subplots(3)
fig.suptitle('Exercitiu1 Sub_pctB')
axs[0].plot(t, xt)
axs[1].plot(t, yt)
axs[2].plot(t, zt)
plt.show()
"""

#Subpct c
freq = 200
t_discret = np.arange(0, 0.03, 1/freq)

pi = np.pi

xt_discret = np.cos(520*pi*t_discret + pi/3)
yt_discret = np.cos(280*pi*t_discret - pi/3)
zt_discret = np.cos(120*pi*t_discret + pi/3)

fig, axs = plt.subplots(3)
fig.suptitle('Exercitiu1 Sub_pctC')
axs[0].plot(t, xt)
axs[0].set_title('x[n]')
axs[1].plot(t, yt)
axs[1].set_title('y[n]')
axs[2].plot(t, zt)
axs[2].set_title('z[n]')
axs[0].stem(t_discret, xt_discret)
axs[1].stem(t_discret, yt_discret)
axs[2].stem(t_discret, zt_discret)
plt.show()




