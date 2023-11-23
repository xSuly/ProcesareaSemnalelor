#Laborator 7 - ex 2 - 23 nov
# Albei Liviu-Andrei === CTI 461

from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt

pi = np.pi

X = misc.face(gray=True)
plt.imshow(X, cmap=plt.cm.gray)
plt.show()

Y = np.fft.fft2(X)
freq_db = 20*np.log10(abs(Y))

plt.imshow(freq_db)
plt.colorbar()
plt.show()

freq_cutoff = 101

Y_cutoff = Y.copy()
Y_cutoff[freq_db > freq_cutoff] = 0
X_cutoff = np.fft.ifft2(Y_cutoff)
X_cutoff = np.real(X_cutoff)    # avoid rounding erros in the complex domain,
                                # in practice use irfft2
plt.imshow(X_cutoff, cmap=plt.cm.gray)
plt.show()