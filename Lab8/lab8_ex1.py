#Laborator 8 - ex 1 - 7 decembrie
# Albei Liviu-Andrei === CTI 461

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

pi = np.pi

N = 1000
x = np.arange(N)

trend = 5*x**2 - 2*x + 10
sezon = np.sin(3*pi*2*x) + np.sin(7*pi*2*x)
variatii_mici = np.random.normal(0, 1, N)

serie_timp = trend + sezon + variatii_mici

fig, axs = plt.subplots(4)
fig.suptitle('Exercitiu1 lab8')
axs[0].plot(x, serie_timp, 'red')
axs[0].set_title('serie timp')
axs[1].plot(x, trend, 'blue')
axs[1].set_title('trend')
axs[2].plot(x, sezon, 'magenta')
axs[2].set_title('sezon')
axs[3].plot(x, variatii_mici, 'green')
axs[3].set_title('variatii mici')
plt.tight_layout()

plt.savefig("Lab8/grafice/lab8_ex1_a.png")
plt.show()



#subpct B

autocorelatie = np.correlate(serie_timp, serie_timp, mode = "full")
corelatie_all = np.arange(0, N)

plt.figure(2)

plt.plot(corelatie_all, autocorelatie[N-1:]) #avem N-1 pentru ca da eroare la dimensiune 1000 si 1999, deci trebuie un N-1 sa fie cu 9 la final

plt.title('Vectorul de autocorelație al seriei de timp') #e asa descendent pentru ca luam jumatatea din dreapta, in stanga urca si in dreapta coboara si se corecteaza din ce in ce mai mult
plt.savefig("Lab8/grafice/lab8_ex1_b.png")
plt.show()



#subpct C
p = 21
model = AutoReg(serie_timp, lags=p)
model_fit = model.fit()

lungime_predictii = 1555
start_predictions = len(serie_timp)
end_predictions = start_predictions + lungime_predictii - 1 #primesc eroare ca are lungime 1556 deci mai scad 1
predictii = model_fit.predict(start = start_predictions, end = end_predictions)

plt.figure(3)
plt.title('Serie + predictii')
plt.plot(x, serie_timp, 'blue')
plt.plot(np.arange(N, N + lungime_predictii), predictii, 'red', label='AR Model')

plt.savefig("Lab8/grafice/lab8_ex1_c.png")
plt.show()


