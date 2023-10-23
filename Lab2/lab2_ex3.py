#Laborator 2 - ex 3 - 19 oct
# Albei Liviu-Andrei === CTI 461

import numpy as np
import matplotlib.pyplot as plt
import sounddevice as snd
import time
import scipy

pi = np.pi

#semnalul de la subpct A, ex2 lab 1
t = np.linspace(0, 4, 1600)
f0_A = 400
sinusoid_A = np.sin(2*pi*f0_A*t)

#semnalul de la subpct B, ex2 lab 1
f0_B = 800 #Hz
t_B = np.linspace(0, 3, 1600)
sinusoid_B = np.sin(2*pi*f0_B*t_B)

#semnalul de la subpct C, ex2 lab 1
f0_C = 240 #Hz
t_C = np.linspace(0, 1, 500) 
sawtooth_C = 2*(t_C*f0_C - np.floor(t_C*f0_C + 1/2))

#semnalul de la subpct D, ex2 lab 1
f0_D = 300 #Hz
t_D = np.linspace(0, 2, 650) 
square_D = np.sign(np.sin(2*pi*t_D*f0_D))

snd.play(square_D)
time.sleep(6.0)
snd.stop()

scipy.io.wavfile.write('square_D.wav', 44100, square_D)
scipy.io.wavfile.read('square_D.wav', 44100) #44100 reprezinta samples per second, wavfile.read citeste doar datele din fisierul .wav, nu ii face si play

