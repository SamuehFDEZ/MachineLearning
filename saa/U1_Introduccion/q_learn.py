# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 22:23:45 2024

@author: Samuel Arteaga López
"""

from random import randint
import numpy as np

# define el grafo del laberinto mediante una matriz de conexión
# -1 significa que no hay enlace entre un lugar y otro por ejemplo
# el estado a no puede ir al b directamente
R = np.array([[-1,-1,-1,-1,0,-1],
              [-1,-1,-1,0,-1,100],
              [-1,-1,-1,0,-1,-1],
              [-1,0,0,-1,0,-1],
              [0,-1,-1,0,-1,100],
              [-1,0,-1,-1,0,100]]).astype(dtype='float32')

# el conocimiento
Q = np.zeros_like(R)

#ratio de aprendizaje
gamma = 0.8

# Inicializar aleatoriamente el estado
estadoInicial = randint(0, 4)

# acciones posibles en este estado

def accionesPosibles(estado):
    filaEstadoActual = R[estado,]
    posibilidades = np.where(filaEstadoActual >= 0)[0]
    return posibilidades

# escoge aleatoriamente
def escogeSigPaso(movimientosDisponibles):
    siguiente = int(np.random.choice(movimientosDisponibles, 1))
    return siguiente

def actualiza(estadoActual, accion, gamma):
    maxIndx = np.where(Q[accion,] == np.max(Q[accion,]))[0]
    if maxIndx.shape[0] > 1:
        maxIndx = int(np.random.choice(maxIndx, size=1))
    else:
        maxIndx = int(maxIndx)
    maxValor = Q[accion, maxIndx]
    # formula del Q learning
    Q[estadoActual, accion] = R[estadoActual, accion] + gamma * maxValor
    pasosDisponibles = accionesPosibles(estadoInicial)
    accion = escogeSigPaso(pasosDisponibles)
    for i in range(100): # entrenar 100 iteraciones
        estadoActual = np.random.randint(0, int(Q.shape[0]))
        pasosDisponibles = accionesPosibles(estadoActual)
        actualiza(estadoActual, accion, gamma)
    print("Q matriz entrenada: \n", Q / np.max(Q) * 100) # Normalizar la Q matriz entrenada

# Testing
estadoActual = 2
pasos = [estadoActual]

while estadoActual != 5:
    idxSigPaso = np.where(Q[estadoActual,] == np.max(Q[estadoActual,]))[0]
    if idxSigPaso.shape[0] > 1:
        idxSigPaso = int(np.random.choice(idxSigPaso, size = 1))
    else:
        idxSigPaso = int(idxSigPaso)
    pasos.append(idxSigPaso)
    estadoActual = idxSigPaso
print("Mejor secuencia de ruta: ", pasos)


