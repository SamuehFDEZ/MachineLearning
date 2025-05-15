# -*- coding: utf-8 -*-
#### apartado a)
import pandas as pd
df = pd.read_csv("../../U03_Recursos/U03_P03_Texto/reseñas_restaurantes.csv", sep=";")
print(df.info())
del(df['puntuación'])       # Borramos puntuaciones
datos = [tuple(x) for x in df.values] # Transformar una lista de tuplas


#### Apartado b)
# Para instalar nltk puedes hacer: pip install nltk==3.9.1
# Si quieres elegir paquetes, corpus, datos individuales ejecuta interactiva:
# import nltk
# nltk.download()
# Y se abre página web donde eliges
import nltk
from tqdm import tqdm
from nltk.tokenize import word_tokenize # Tokeniza palabras
from nltk.corpus import stopwords    # stop-words, palabras comunes
nltk.download('punkt')              # Es un tokenizer de nltk en inglés y español
nltk.download('stopwords')

spanish_sw = set(stopwords.words('spanish'))

from nltk.stem.porter import PorterStemmer # Stemmer para eliminar variedad
porter = PorterStemmer()

def normaliza(mensajes):
    """
    Preprocesa los datos de texto
    Entradas: mensajes con formato [(texto, etiqueta)]
    Salidas: mensajes con formato [(texto, label), (texto, label)...]
    """
    for idx, msj in enumerate(tqdm(mensajes)):
        tokens = word_tokenize(msj[0].lower(), language="spanish") # Paso a minúsculas y tokenizamos
        filtrado = [palabra for palabra in tokens
                    if (len(palabra) > 2)
                    and (not palabra in spanish_sw)]
        stemmed = " ".join([porter.stem(p) for p in filtrado])
        mensajes[idx] = (stemmed, msj[1])
    return mensajes

X = normaliza(datos) # Normalizamos el texto de los comentarios


#### Apartado c)
y = [y[1] for y in X]
n_positivos = y.count('positivo')
n_total = len(y)
r_positivos = n_positivos / n_total
r_negativos = 1 - r_positivos
print(f"\nReseñas en total ({n_total}) Positivos ({(100.0 * r_positivos):.4f}%)")
from sklearn.model_selection import train_test_split

# Separamos texto y etiquetas
X_text = [x[0] for x in X]
y_labels = [x[1] for x in X]

# División 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X_text, y_labels, test_size=0.2, random_state=675)

# Verificamos proporciones
from collections import Counter

def proporciones(y):
    total = len(y)
    conteo = Counter(y)
    for etiqueta in conteo:
        print(f"{etiqueta}: {conteo[etiqueta]} ({100.0 * conteo[etiqueta] / total:.2f}%)")

print("\nProporciones en train:")
proporciones(y_train)

print("\nProporciones en test:")
proporciones(y_test)

#### Apartado d)
def palabras_diferentes(datos):
    """Devuelve una lista con todas las palabras únicas de las frases"""
    todas = []
    for (texto, sentimiento) in datos:
        todas.extend(nltk.word_tokenize(texto, language="spanish"))
    return list(set(todas))

diccionario = palabras_diferentes(X) # Creamos el diccionario de palabras

def extrae_caracteristicas(texto):
    """
    Crea el conjunto de entrenamiento del clasificador
    1: Para cada palabra del diccionario
    3: Escribe {'contains(palabra)':True,...} si aparece palabra en texto
       Escribe {'contains(palabra)':False,...} si no aparece
    """
    palabras_del_texto = set(texto.split())
    caracteristicas = {}
    for palabra in diccionario:
        caracteristicas[f'contains({palabra})'] = (palabra in palabras_del_texto)
    return caracteristicas

# Creamos las estructuras de datos para el clasificador con NLTK
X_train_nltk = nltk.classify.apply_features(extrae_caracteristicas, list(zip(X_train, y_train)))
X_test_nltk = nltk.classify.apply_features(extrae_caracteristicas, list(zip(X_test, y_test)))


# Apartado e)
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

print('Entrenando Naive Bayes')
nb = nltk.classify.NaiveBayesClassifier.train(X_train_nltk)
print('Entrenando Naive Bayes Multinomial')
nbm = nltk.classify.SklearnClassifier(MultinomialNB()).train(X_train_nltk)
print('Entrenando Bernoulli Naive Bayes')
nbb = nltk.classify.SklearnClassifier(BernoulliNB()).train(X_train_nltk)
print('Entrenando Regresión Logística')
rl = nltk.classify.SklearnClassifier(LogisticRegression(solver='lbfgs', max_iter=1000)).train(X_train_nltk)
print('Entrenando Support Vector Machine')
svm = nltk.classify.SklearnClassifier(SVC(kernel='linear')).train(X_train_nltk)
clasificadores = {
    'Naive Bayes': nb,
    'Naive Bayes Multinomial': nbm,
    'Naive Bayes Bernoulli': nbb,
    'Regresion Logistica': rl,
    'Support Vector Machine': svm
}

#### Apartado f)
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

evaluacion = list()
for k, v in clasificadores.items():
    print(f'Evaluando Modelo: {k}')
    # Obtengo las predicciones
    predicciones = [v.classify(texto[0]) for texto in X_test_nltk]
    modelo = {}
    modelo['nombre'] = k
    modelo['accuracy'] = accuracy_score(y_test, predicciones)
    modelo['precision'] = precision_score(y_test, predicciones, average='weighted')
    modelo['recall'] = recall_score(y_test, predicciones, average='weighted')
    modelo['f1'] = f1_score(y_test, predicciones, average='weighted')
    evaluacion.append(modelo)

''' ENTREGA 12:
 Crea el programa y lo ejecutas y mira las métricas de los diferentes clasificadores. Pasa 
una captura de pantalla de los resultados obtenidos 
'''

# Pasamos los resultados a un DataFrame para visualizarlos mejor
df = pd.DataFrame.from_dict(evaluacion)
df.set_index("nombre", inplace=True)
print(df)

'''
ENTREGA 13:
 Escribe una evaluación razonada de los resultados (cuál te quedarías, el motivo…).
'''
'''
Tras comparar los distintos clasificadores, el modelo que mejores métricas globales 
ha obtenido ha sido el Naive Bayes, normal y multinominal, igualando los dos en accuracy. 
Me quedaría con Naive Bayes, ya que ofrece el mejor rendimiento general,en comparación a los otros 
especialmente si el conjunto de datos sigue creciendo y se vuelve más variado. 
En tareas de texto, Naive Bayes con kernel lineal suele destacar en clasificación binaria 
como en este caso (positivo/negativo).
'''


'''
 ENTREGA 14: 
Repite el ejercicio pero ahora entrena los clasificadores con el fichero original, es decir, 
antes de que tú y tus compañeros hayan añadido nuevos datos (los datos originales serán las primeras 
51 filas de datos). Muestra captura de pantalla de los resultados como en la entrega 17.
'''
# Leer solo las primeras 51 filas
df = pd.read_csv("../../U03_Recursos/U03_P03_Texto/reseñas_restaurantes.csv", sep=";").head(51)
del(df['puntuación'])
datos = [tuple(x) for x in df.values]

#### Apartado b)
# Para instalar nltk puedes hacer: pip install nltk==3.9.1
# Si quieres elegir paquetes, corpus, datos individuales ejecuta interactiva:
# import nltk
# nltk.download()
# Y se abre página web donde eliges
import nltk
from tqdm import tqdm
from nltk.tokenize import word_tokenize # Tokeniza palabras
from nltk.corpus import stopwords    # stop-words, palabras comunes
nltk.download('punkt')              # Es un tokenizer de nltk en inglés y español
nltk.download('stopwords')

spanish_sw = set(stopwords.words('spanish'))

from nltk.stem.porter import PorterStemmer # Stemmer para eliminar variedad
porter = PorterStemmer()

def normaliza(mensajes):
    """
    Preprocesa los datos de texto
    Entradas: mensajes con formato [(texto, etiqueta)]
    Salidas: mensajes con formato [(texto, label), (texto, label)...]
    """
    for idx, msj in enumerate(tqdm(mensajes)):
        tokens = word_tokenize(msj[0].lower(), language="spanish") # Paso a minúsculas y tokenizamos
        filtrado = [palabra for palabra in tokens
                    if (len(palabra) > 2)
                    and (not palabra in spanish_sw)]
        stemmed = " ".join([porter.stem(p) for p in filtrado])
        mensajes[idx] = (stemmed, msj[1])
    return mensajes

X = normaliza(datos) # Normalizamos el texto de los comentarios


#### Apartado c)
y = [y[1] for y in X]
n_positivos = y.count('positivo')
n_total = len(y)
r_positivos = n_positivos / n_total
r_negativos = 1 - r_positivos
print(f"\nReseñas en total ({n_total}) Positivos ({(100.0 * r_positivos):.4f}%)")
from sklearn.model_selection import train_test_split

# Separamos texto y etiquetas
X_text = [x[0] for x in X]
y_labels = [x[1] for x in X]

# División 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X_text, y_labels, test_size=0.2, random_state=675)

# Verificamos proporciones
from collections import Counter

def proporciones(y):
    total = len(y)
    conteo = Counter(y)
    for etiqueta in conteo:
        print(f"{etiqueta}: {conteo[etiqueta]} ({100.0 * conteo[etiqueta] / total:.2f}%)")

print("\nProporciones en train:")
proporciones(y_train)

print("\nProporciones en test:")
proporciones(y_test)

#### Apartado d)
def palabras_diferentes(datos):
    """Devuelve una lista con todas las palabras únicas de las frases"""
    todas = []
    for (texto, sentimiento) in datos:
        todas.extend(nltk.word_tokenize(texto, language="spanish"))
    return list(set(todas))

diccionario = palabras_diferentes(X) # Creamos el diccionario de palabras

def extrae_caracteristicas(texto):
    """
    Crea el conjunto de entrenamiento del clasificador
    1: Para cada palabra del diccionario
    3: Escribe {'contains(palabra)':True,...} si aparece palabra en texto
       Escribe {'contains(palabra)':False,...} si no aparece
    """
    palabras_del_texto = set(texto.split())
    caracteristicas = {}
    for palabra in diccionario:
        caracteristicas[f'contains({palabra})'] = (palabra in palabras_del_texto)
    return caracteristicas

# Creamos las estructuras de datos para el clasificador con NLTK
X_train_nltk = nltk.classify.apply_features(extrae_caracteristicas, list(zip(X_train, y_train)))
X_test_nltk = nltk.classify.apply_features(extrae_caracteristicas, list(zip(X_test, y_test)))


# Apartado e)
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

print('Entrenando Naive Bayes')
nb = nltk.classify.NaiveBayesClassifier.train(X_train_nltk)
print('Entrenando Naive Bayes Multinomial')
nbm = nltk.classify.SklearnClassifier(MultinomialNB()).train(X_train_nltk)
print('Entrenando Bernoulli Naive Bayes')
nbb = nltk.classify.SklearnClassifier(BernoulliNB()).train(X_train_nltk)
print('Entrenando Regresión Logística')
rl = nltk.classify.SklearnClassifier(LogisticRegression(solver='lbfgs', max_iter=1000)).train(X_train_nltk)
print('Entrenando Support Vector Machine')
svm = nltk.classify.SklearnClassifier(SVC(kernel='linear')).train(X_train_nltk)
clasificadores = {
    'Naive Bayes': nb,
    'Naive Bayes Multinomial': nbm,
    'Naive Bayes Bernoulli': nbb,
    'Regresion Logistica': rl,
    'Support Vector Machine': svm
}

#### Apartado f)
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

evaluacion = list()
for k, v in clasificadores.items():
    print(f'Evaluando Modelo: {k}')
    # Obtengo las predicciones
    predicciones = [v.classify(texto[0]) for texto in X_test_nltk]
    modelo = {}
    modelo['nombre'] = k  # <- Añadir esta línea
    modelo['accuracy'] = accuracy_score(y_test, predicciones)
    modelo['precision'] = precision_score(y_test, predicciones, average='weighted')
    modelo['recall'] = recall_score(y_test, predicciones, average='weighted')
    modelo['f1'] = f1_score(y_test, predicciones, average='weighted')
    evaluacion.append(modelo)

# Pasamos los resultados a un DataFrame para visualizarlos mejor
df = pd.DataFrame.from_dict(evaluacion)
df.set_index("nombre", inplace=True)
print(df)

'''
 ENTREGA 15
 : Valora si al añadir más datos se mejora la eficiencia de los clasificadores, a cuáles les 
afecta más el aumento en la cantidad de datos, etc. '''

'''
Dado que no contamos con demasiados datos y sobretodo son aleatorios, que no se ajustan a 
ningun patron definido, de ello los resultados son iguales.
'''









