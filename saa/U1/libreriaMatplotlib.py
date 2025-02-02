import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 10, 50)
senos = np.sin(x)
plt.plot(x, senos)
plt.show()

# Indicando una figura en vez de una línea recta
x = np.linspace(0, 10, 50)
senos = np.sin(x)
plt.plot(x, senos, "o") # Si no se indica por defecto es "-" una línea
plt.show()

x = np.linspace(0, 10, 50)
senos = np.sin(x)
cosenos = np.cos(x)
plt.plot(x, senos, "-b", x, senos, "ob", x, cosenos, "-r", x, cosenos, "or")
plt.xlabel("Este es el eje x!") # Etiqueta del eje X
plt.ylabel("Este es el eje y!") # Etiqueta del eje Y
plt.title("Mis primeras gráficas") # Títul

