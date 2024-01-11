#Laborator 10 - ex 1 - 11 ianuarie
# Albei Liviu-Andrei === CTI 461

import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
media = 0 #mu
varianta = 0.2 #sigma

x = np.linspace(-5*varianta,5*varianta,100)
dG = (1/varianta* np.sqrt(2*pi))*np.exp(-(x-media)**2/(2*varianta))

plt.figure(figsize=(12, 6))
plt.plot(x, dG)
plt.xlabel('X')
plt.ylabel('gamma')
plt.title('Exercitiu1 lab10')
plt.legend(loc='lower right')
plt.show()

x = np.array([[1],[1]])

mat_covarianta = np.array([[1, 3/5], [3/5, 2]])
medie_wikipedia = np.array([[0], [0]])

inv_mat_covarianta = np.linalg.inv(mat_covarianta)

det_mat_covarianta = np.linalg.det(mat_covarianta)

dG_bidimensionala = (1/(2*pi)**2)*det_mat_covarianta**(-1/2)*np.exp(-0.5*((x-medie_wikipedia).T@inv_mat_covarianta@(x-medie_wikipedia)))

plt.scatter(x, dG_bidimensionala)

plt.title('dG bidimensionala')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

