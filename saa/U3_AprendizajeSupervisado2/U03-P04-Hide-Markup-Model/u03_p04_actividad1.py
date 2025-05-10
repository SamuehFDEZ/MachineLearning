# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

fichero_audio = r"../U03_Recursos/U03_P04_Audios/0-SAMARTLOP-0.m4a"
frecuencia, audio = wavfile.read(fichero_audio)
duracion = round(audio.shape[0] / float(frecuencia), 3)
print ('\nMuestras:', audio.shape)
print ('Tipo de dato de cada muestra:', audio.dtype)
print ('Duración:', duracion, 'segundos')

audio = audio / 2.**15 # Normalizar
audio1 = audio[:240]   # Dibujar 240 muestras
x_val = np.arange(0, len(audio1), 1) / float(frecuencia)
x_val *= 1000
plt.plot(x_val, audio1, color='black')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Amplitud')
plt.title('Señal de Audio ' + fichero_audio)
plt.show()

''' ENTREGA 2:
 Copia el código y lo modificas para que coja tu fichero cuando pronuncias el cero. 
a) ¿La onda de tu gráfico es exactamente igual que la de mi pronunciación o tiene diferencias?
 b)
 Entrega el gráfico de tu onda de audio junto a la mía.'''


audio_transformado = np.fft.fft(audio) # FFT
mitad = int(np.ceil((len(audio) + 1) / 2.0))
audio_transformado = abs(audio_transformado[0:mitad])
audio_transformado **= 2

potencia = 20 * np.log10(audio_transformado + 1)
x_val = np.arange(0, mitad, 1) * (frecuencia / len(audio)) / 1000.0
plt.figure()
plt.plot(x_val, potencia, color='black')
plt.xlabel('Frecuencia (kHz)')
plt.ylabel('Potencia (dB)')
plt.show()

duracion = 3
frecuencia_tono = 587
min_val = -2 * np.pi
max_val = 2 * np.pi
t = np.linspace(min_val, max_val, duracion * frecuencia)
audio = np.sin(2 * np.pi * frecuencia_tono * t)
ruido = 0.4 * np.random.rand(duracion * frecuencia)
audio += ruido
escala = pow(2, 15) - 1
audio_normalizado = audio / np.max(np.abs(audio))
audio_escalado = np.int16(audio_normalizado * escala)
x_val = np.arange(0, len(audio), 1) / float(frecuencia)
x_val *= 1000
plt.plot(x_val[:60], audio[:60], color='red')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Amplitud')
# Guardar el audio a un fichero
fichero_salida = './generado.wav'
wavfile.write(fichero_salida, frecuencia, audio_escalado)
plt.title('Señal de Audio Generado ' + fichero_salida)
plt.show()


# Extraer las características de un audio
from librosa.feature import mfcc
import librosa

audio, frecuencia = librosa.load(fichero_audio)
caracteristicas_mfcc = mfcc(sr=frecuencia, y=audio)
print("Número de frames =", caracteristicas_mfcc.shape[1])
print('Longitud de cada característica =', caracteristicas_mfcc.shape[0])
# Dibujarlas
caracteristicas_mfcc = caracteristicas_mfcc.T
plt.matshow(caracteristicas_mfcc)
plt.title('MFCC')
plt.show()

''' ENTREGA 3:
 Haz lo mismo con tu fichero de audio del número cero y 
entregas captura del resultado de la ejecución.'''




# -*- coding: utf-8 -*-
import itertools
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from hmmlearn import hmm
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank, delta
# import librosa                      # Otra posibilidad de leer audio: audio, fm = librosa.read(fichero_audio)
# from librosa.feature import mfcc    # Otra posibilidad de extraer escalas mel del audio
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings("ignore")

########## PARÁMETROS PARA DEFINIR EL DATASET ##########
carpeta = "./audios/"
n_caracteristicas = 10         # Num. de características mel de cada trozo de audio
tama_caracteristicas = 10        # Cantidad de datos de cada característica
valor_imputado = 0.1           # Mecanismo de seguridad para no alimentar HMM con valores NaN
mucha_informacion = True        # Mostrar progreso de cada acción paso a paso
ratio_train = 0.1             # Porcentaje de ejemplos para train ente 0.0 y 1.0
########## HIPERPARÁMETROS DE LOS MODELOS: ##########
# Hay dos posibles modelos:
# - hmm.GMMHMM()                Gaussiano Mixture con HMM. La emisión de un símbolo se modela como una
#                               distribución aleatoria que es una mezcla de otras distribuciones.
# - hmm.GaussianHMM()           Gaussiano con HMM. La emisión obedece a una única distribución.
# Aunque tienen más, solamente usaremos estos:
n_estados_ocultos = 1          # Estados ocultos usados en el HMM
n_iteraciones = 10             # Iteraciones en el algoritmo (Viterbi por defecto)



def definir_dataset(ruta_audios=carpeta):
    ficheros = sorted(os.listdir(ruta_audios))
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    datos = dict()
    n = len(ficheros)
    digito_actual = ""
    for i in range(n):
        if not ficheros[i].lower().endswith('.wav'):
            continue
        digito = ficheros[i][0]  # digito-nombre.wav
        if mucha_informacion and digito_actual != digito:
            digito_actual = digito
            print("Procesando ficheros del dígito " + digito)
        carac = extraer_caracteristicas(os.path.join(ruta_audios, ficheros[i]))
        if digito not in datos.keys():
            datos[digito] = [carac]
            x_train.append(carac)
            y_train.append(digito)
        else:
            if np.random.rand() < ratio_train: # Añadirlo a train o a test
                x_test.append(carac)
                y_test.append(digito)
            else:
                datos[digito].append(carac)
                x_train.append(carac)
                y_train.append(digito)
    return x_train, y_train, x_test, y_test, datos



def extraer_caracteristicas(ruta_audio):
    # audio, frec_muestreo = librosa.load(ruta_audio)
    # mfcc_carac = mfcc(y=audio, sr=frec_muestreo, n_mfcc=n_caracteristicas)
    frec_muestreo, audio = wavfile.read(ruta_audio)
    mfcc_carac = mfcc(audio, samplerate=frec_muestreo, numcep=n_caracteristicas, nfilt=tama_caracteristicas)
    if mucha_informacion:
        print(f"[{ruta_audio}] Dimensiones de mfcc {mfcc_carac.shape}")
    return mfcc_carac

# El entrenamiento lo realizaremos en otra función de Python:
def entrenar_modelo(datos):
    hmm_aprendido = dict()
    for label in datos.keys():
        # modelo = hmm.GMMHMM(n_components=n_estados_ocultos, n_iter=n_iteraciones)
        modelo = hmm.GaussianHMM(n_components=n_estados_ocultos, n_iter=n_iteraciones, verbose=mucha_informacion)
        caracteristica = None
        for cada_caracteristica in datos[label]:
            cada_caracteristica = np.nan_to_num(cada_caracteristica, nan=valor_imputado)
            if caracteristica is None:
                caracteristica = cada_caracteristica
            else:
                caracteristica = np.vstack((caracteristica, cada_caracteristica))
        obj = modelo.fit(caracteristica)
        if mucha_informacion:
            print("***** Modelo de", label)
            print("Prob. de inicio:", obj.startprob_)
            print("Matriz de transición:\n", obj.transmat_)
        hmm_aprendido[label] = obj
    return hmm_aprendido


def hacer_prediccion(datos_test, entrenado):
    label_predicha = []
    nombres = []
    # predecir una lista de test
    if type(datos_test) == type([]):
        for test in datos_test:
            scores = []
            for nodo in entrenado.keys():
                scores.append(entrenado[nodo].score(test))
                nombres.append(nodo)
            label_predicha.append(scores.index(max(scores)))
    else:
        scores = []
        for nodo in entrenado.keys():
            scores.append(entrenado[nodo].score(datos_test))
            nombres.append(nodo)
        label_predicha.append(scores.index(max(scores)))
    return nombres[label_predicha[0]]

# En los datos test dibujaremos la matriz de confusión con esta función:
def plot_matriz_confusion(cm, clases, normaliza=False, titulo='Matriz de Confusión', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(titulo)
    plt.colorbar()
    tick_marks = np.arange(len(clases))
    plt.xticks(tick_marks, clases, rotation=45)
    plt.yticks(tick_marks, clases)

    fmt = '.2f' if normaliza else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Label real')
    plt.xlabel('Label predicha')
    plt.show()


def informe(y_test, y_pred, mostrar_grafico=True):
    print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
    print("-" * 40)
    print("Informe de clasificación:\n\n", classification_report(y_test, y_pred))
    print("-" * 40)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("-" * 40 + "\n")
    if mostrar_grafico:
        plot_matriz_confusion(confusion_matrix(y_test, y_pred), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# El programa que llama a todas estas funciones está dividido en tres partes. Una primera parte en la que preparamos el dataset:

# ===== PASO 1: Definir el Dataset =====
print("===== PASO 1: Definir el Dataset =====")
x_train, y_train, x_test, y_test, datos = definir_dataset()
print("Datos de entrenamiento:", len(x_train))
print("Datos de test:", len(x_test))
print("Diccionario de datos contiene datos de:", datos.keys())

# ===== PASO 2: Entrenar el modelo =====
print("\n===== PASO 2: Entrenar el modelo =====")
hmm_aprendido = entrenar_modelo(datos)

# Guardar modelo
with open("modelo_aprendido.pkl", "wb") as fichero:
    pickle.dump(hmm_aprendido, fichero)
print("Entrenamiento realizado...")

# Parte donde realizamos el test cargando primero el modelo almacenado.
# Esto se haría en caso de usarlo en producción:

# ===== PASO 3: Usar el modelo para predecir =====
print("\n===== PASO 3: Usar el modelo para predecir =====")
# Leer modelo de disco si queremos usarlo tras crearlo
with open("u03_p04_modelo_hmm.pkl", "rb") as fichero:
    hmm = pickle.load(fichero)
ficheros = sorted(os.listdir(carpeta))
tot_test = 0
tot_train = 0
n = len(x_test)
m = len(x_train)
pred_test = []
pred_train = []
for i in range(m):
    y_pred = hacer_prediccion(x_train[i], hmm)
    if y_pred == y_train[i]:
        tot_train += 1
    pred_train.append(y_pred)

for i in range(n):
    y_pred = hacer_prediccion(x_test[i], hmm)
    if y_pred == y_test[i]:
        tot_test += 1
    pred_test.append(y_pred)

informe(y_train, pred_train)
print("###################################### TRAIN ACCURACY ######################################")
print(tot_train/m)
print("###################################### TEST ACCURACY #######################################")
print(tot_test/n)


''' ENTREGA 4:
 Entrega:
 a) El código del programa Python.
 b)
 Captura de pantalla del resultado de una ejecución. '''






''' ENTREGA 5:
 Entrega:
 a) Captura de pantalla del resultado de la ejecución con los mejores parámetros.
 b)
 Valor establecido para:
 • n_características: 
• valor_imputado: 
• ratio_de_train: 
• modelo usado: 
( ) GMMHMM.        ( ) GaussianHMM. 
• n_estados_ocultos: 
• n_iteraciones: 
c)
 Valor conseguido de (ver figuras para valores de referencia):
 • Accuracy Test (0.81):   
• Media de precisión (0.7):   
• Media de recall (0.81):  
• Media de F1-score (0.75):   
• support(120):   
d) Gráfico de la matriz de confusión del test
 e) Fichero del mejor modelo generado (
 .pkl)'''

n_caracteristicas = 13        # Más características mel (por defecto en MFCC)
tama_caracteristicas = 26     # Más filtros mel para mayor detalle espectral
valor_imputado = 0.01         # Valor más pequeño para evitar alterar el modelo
ratio_train = 0.8             # Mayor proporción de datos de entrenamiento

n_estados_ocultos = 4         # Más estados ocultos da mayor flexibilidad al modelo
n_iteraciones = 30            # Más iteraciones para mayor convergencia


modelo = hmm.GMMHMM(n_components=n_estados_ocultos, n_iter=n_iteraciones)
