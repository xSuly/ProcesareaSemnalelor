#Laborator 2 - ex 5 - 19 oct
# Albei Liviu-Andrei === CTI 461

import numpy as np
import matplotlib.pyplot as plt
import sounddevice as snd
import time

pi = np.pi

f0_A = 400
t = np.linspace(0, 4, 1600)

f0_B = 750
sinusoid_A = np.sin(2*pi*f0_A*t)
sinusoid_B = np.sin(2*pi*f0_B*t)

vector_combinatie = np.concatenate((sinusoid_A, sinusoid_B))

snd.play(vector_combinatie, samplerate= 7500) #setez samplerate la 7500 pentru a auzi in slowmotion sunetul si sa imi dau seama de rezultate
time.sleep(3.0)
snd.stop()

#Notita: daca doar frecventa difera, sunetul redat prin combinarea celor 2 sinusoide este acelasi, continua identic sau cel putin asa da impresia
#Daca schimbam formula, se observa o diferenta majora iar sunetul nu esti continuu




