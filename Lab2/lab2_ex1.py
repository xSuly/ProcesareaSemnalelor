#Laborator 2 - ex 1 - 19 oct
# Albei Liviu-Andrei === CTI 461

import numpy as np
import matplotlib.pyplot as plt


x_f0 = 1.25 # frecventa sinusoidei /// oscilatii intr-o secunda
t = np.linspace(0, 3)

pi = np.pi
xA = 1 # amplitudinea
# f = 1/T
T = 400

x_t = 5 #timpul in secunde 
x_faza = 0 #faza

x = xA * np.sin(2 * pi * x_f0 * t + x_faza)

y = xA * np.cos(2 * pi * x_f0 * t + x_faza - pi/2 )

fig, axs = plt.subplots(2)
fig.suptitle('Exercitiu1')
axs[0].plot(t, x)
axs[0].set_title('Sinusoidal')
axs[1].plot(t, y)
axs[1].set_title('Tip cosinus')
plt.tight_layout()
plt.show()