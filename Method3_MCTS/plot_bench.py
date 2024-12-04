import numpy as np 
import matplotlib.pyplot as plt 

y = [0, 452, 402, 43888, 34809, 34825, 21231, 18401, 337614, 299769, 143075, 137567, 1425242, 608967, 134754, 30259, 3387, 169, 25, 3]
x = np.arange(1, len(y)+1)


plt.plot(x[:7],y[:7])
plt.title("Nombres d'états visités à chaque tour")
plt.xlabel("Numéro du tour")
plt.ylabel("Nombres d'états visités")
plt.grid()
plt.legend()
plt.show()

plt.plot(x,y)
plt.title("Nombres d'états visités à chaque tour")
plt.xlabel("Numéro du tour")
plt.ylabel("Nombres d'états visités")
plt.grid()
plt.legend()
plt.show()