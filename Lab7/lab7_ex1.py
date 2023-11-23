#Laborator 7 - ex 1 - 23 nov
# Albei Liviu-Andrei === CTI 461

from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt

pi = np.pi

X = misc.face(gray=True)
#plt.imshow(X, cmap=plt.cm.gray)
#plt.show()

#ex1 subpct A

n1 = 10
n2 = 20

x_A = np.zeros((n1,n2))
for i in range(n1):
    for j in range(n2):
        x_A[i, j] = np.sin(2*pi*i + 3*pi*j)

plt.imshow(x_A)
plt.show()

Y = np.fft.fft2(x_A)
freq_db = 20*np.log10(abs(Y))

plt.imshow(freq_db)
plt.colorbar()
plt.show()

#ex1 subpct B

x_B = np.zeros((n1, n2))
for i in range(n1):
    for j in range(n2):
        x_B[i, j] = np.sin(4*pi*i) + np.cos(6*pi*j)

plt.imshow(x_B, cmap=plt.cm.gray)
plt.show()

Y_B = np.fft.fft2(x_B)
freq_db_B = 20*np.log10(abs(Y_B))

plt.imshow(freq_db_B)
plt.colorbar()
plt.show()

#ex1 subpct C ~~~ luam procesul invers acum
Y_C = np.zeros((n1, n2))
Y_C[0][5] = 1
Y_C[0][n2 - 5] = 1

freq_db_C = 20 * np.log10(abs(Y_C))

plt.imshow(freq_db_C)
plt.colorbar()
plt.show()

x_C = np.fft.ifft2(Y_C)

plt.imshow(x_C.real)
plt.show()

#ex1 subpct D 
Y_D = np.zeros((n1, n2))
Y_D[5][0] = 1
Y_D[n1 - 5][0] = 1

freq_db_D = 20 * np.log10(abs(Y_D))

plt.imshow(freq_db_D)
plt.colorbar()
plt.show()

x_D = np.fft.ifft2(Y_D)

plt.imshow(x_D.real)
plt.show()

#ex1 subpct E
Y_E = np.zeros((n1, n2))
Y_E[5][5] = 1
Y_E[n1 - 5][n2 - 5] = 1

freq_db_E = 20 * np.log10(abs(Y_E))

plt.imshow(freq_db_E)
plt.colorbar()
plt.show()

x_E = np.fft.ifft2(Y_E)

plt.imshow(x_E.real)
plt.show()