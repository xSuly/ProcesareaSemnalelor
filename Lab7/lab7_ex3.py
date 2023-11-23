#Laborator 7 - ex 3 - 23 nov
# Albei Liviu-Andrei === CTI 461

from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt

pi = np.pi

X = misc.face(gray=True)
#plt.imshow(X, cmap=plt.cm.gray)
#plt.show()

Y = np.fft.fft2(X)
freq_db = 20*np.log10(abs(Y))

#plt.imshow(freq_db)
#plt.colorbar()
#plt.show()

pixel_noise = 300

noise = np.random.randint(-pixel_noise, high=pixel_noise+1, size=X.shape)
X_noisy = X + noise
plt.imshow(X, cmap=plt.cm.gray)
plt.title('Original')
plt.show()
plt.imshow(X_noisy, cmap=plt.cm.gray)
plt.title('Noisy')
plt.show()