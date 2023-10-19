#Laborator 2 - ex 2 - 19 oct
# Albei Liviu-Andrei === CTI 461

import numpy as np
import matplotlib.pyplot as plt
#import sounddevice as snd


t = np.linspace(0, 3, 350)



pi = np.pi

x_A = 1 #amplitudine unitara
x_f0 = 0.65 #frecventa sinusoidei
x_faza = 0 #faza sinusoidei
y_faza = 0.55
z_faza = 0.85
w_faza = 0.25
x = x_A * np.sin(2 * pi * x_f0 * t + x_faza)
y = x_A * np.sin(2 * pi * x_f0 * t + y_faza)
z = x_A * np.sin(2 * pi * x_f0 * t + z_faza)
w = x_A * np.sin(2 * pi * x_f0 * t + w_faza)

zet = np.random.normal(0, 1, len(t))

fig, axs = plt.subplots(2)
fig.suptitle('Exercitiu2')


SNR = [0.1, 1, 10, 100]
for i in SNR:
    gamma = np.sqrt(np.linalg.norm(x)**2/(i * np.linalg.norm(zet)**2))
    axs[0].plot(t, x + gamma * zet)
    #snd.play(x+gamma*zet)
axs[0].set_title('Sinusoidal_random_1')
#for i in SNR:
 #   gamma2 = np.sqrt(np.linalg.norm(y)**2/(i * np.linalg.norm(zet)**2))
  #  axs[1].plot(t, y + gamma * zet)
#axs[1].set_title('Sinusoidal_random_2')



plt.tight_layout()
plt.show()

#plt.figure(figsize=(12, 6))
#plt.plot(t, x, label='sinusoida_1')
#plt.plot(t, y, label='sinusoida_2')
#plt.plot(t, z, label='sinusoida_3')
#plt.plot(t, w, label='sinusoida_4')
#plt.xlabel('Timp')
#plt.ylabel('Amplitudine')
#plt.title('Exercitiu2')
#plt.legend(loc='lower right')
#plt.show()