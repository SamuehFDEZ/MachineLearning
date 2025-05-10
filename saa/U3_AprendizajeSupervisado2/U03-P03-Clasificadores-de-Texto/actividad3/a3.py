# -*- coding: utf-8 -*-
import pandas as pd

sms_spam = pd.read_csv('../../U03_Recursos/U03_P03_Texto/SMSSpamCollection.csv', sep='\t', header=None, names=['Label', 'SMS'])
print("== Dimensiones: ", sms_spam.shape)
print("== Primeros 5 ejemplos:\n", sms_spam.head() )
print("== Información de las columnas:")
print(sms_spam.info())

print("== Porcentajes de spam y ham:")
print( sms_spam['Label'].value_counts(normalize=True) )

# Dividir en train y test
datos = sms_spam.sample(frac=1, random_state=1) # Aleatorizar dataset
indices = round( len(datos) * 0.8 )                 # Calcula índices división
train = datos[:indices].reset_index(drop=True)
test = datos[indices:].reset_index(drop=True)
print("== Dimensiones de train:", train.shape)
print("== Dimensiones de test:", test.shape)
print("== Porcentajes de spam en datos train:")
print( train['Label'].value_counts(normalize=True) )
print("== Porcentajes de spam en datos test:")
print( test['Label'].value_counts(normalize=True) )

import re
import string

# Mostrar las dos primeras filas ANTES de limpiar
print("ANTES DE LA LIMPIEZA:")
print(train['SMS'].head(2))

# Limpieza de datos
train['SMS'] = train['SMS'].str.replace('[{}]'.format(re.escape(string.punctuation)), ' ', regex=True)  # Quitar signos
train['SMS'] = train['SMS'].str.lower()  # Convertir a minúsculas

# Mostrar las dos primeras filas DESPUÉS de limpiar
print("\nDESPUÉS DE LA LIMPIEZA:")
print(train['SMS'].head(2))


'''
 ENTREGA 8:
 Modifica este trozo de código y añade dos sentencias para que muestre las dos primeras 
filas de 
train
 antes y después de aplicar la limpieza de datos como se ve en la figura de abajo para 
comprobar que efectivamente eliminas los signos de puntuación y conviertes a minúsculas. Si no 
consigue hacerlo intenta algunas de estas modificaciones:
 • Importa
 re (expresiones regulares) y string y sustituye la línea 27 por esta: 
re.sub('[%s]' % re.escape(string.punctuation), ' ', train['SMS'].str)
 string y sustituye 
• Importa  la línea 27 por esta: 
train['SMS'].str.replace('[{}]'.format(string.punctuation), ' ', regex=True)
 • A la función replace() de la línea 27 le añades el parámetro: 
train['SMS'] = train['SMS'] = regex=True
'''

# Crear el vocabulario
train['SMS'] = train['SMS'].str.split()
vocabulario = []
for sms in train['SMS']:
    for palabra in sms:
        vocabulario.append(palabra)
vocabulario = list(set(vocabulario))
print(f"== Hay {len(vocabulario)} palabras distintas en los mensajes de train.")

palabra_contadores_por_sms = {'secret': [2, 1, 1],
                              'prize': [2, 0, 1],
                              'claim': [1, 0, 1],
                              'now': [1, 0, 1],
                              'coming': [0, 1, 0],
                              'to': [0, 1, 0],
                              'my': [0, 1, 0],
                              'party': [0, 1, 0],
                              'winner': [0, 0, 1]
                              }

palabras = pd.DataFrame(palabra_contadores_por_sms)
print(palabras.head())

train = pd.concat([train, palabras], axis=1)
print( train.head())

# Calcular el modelo
sms_spam = train[train['Label'] == 'spam']
sms_ham = train[train['Label'] == 'ham']

p_spam = len(sms_spam) / len(train)
p_ham = len(sms_ham) / len(train)

n_spam = sms_spam['SMS'].apply(len)
n_spam = n_spam.sum()
n_ham = sms_ham['SMS'].apply(len)
n_ham = n_ham.sum()
n_vocabulary = len(vocabulario)
alfa = 1

# Inicializar y calcular los parámetros
param_spam = {p: 0 for p in vocabulario}
param_ham = {p: 0 for p in vocabulario}

for palabra in vocabulario:
    if palabra in sms_spam.columns:
        n_wi_spam = sms_spam[palabra].sum()
    else:
        n_wi_spam = 0
    p_wi_spam = (n_wi_spam + alfa) / (n_spam + alfa * n_vocabulary)
    param_spam[palabra] = p_wi_spam

    if palabra in sms_ham.columns:
        n_wi_ham = sms_ham[palabra].sum()
    else:
        n_wi_ham = 0
    p_wi_ham = (n_wi_ham + alfa) / (n_ham + alfa * n_vocabulary)
    param_ham[palabra] = p_wi_ham


import re

def clasifica(mensaje):
    mensaje = re.sub('\W', ' ', mensaje)
    mensaje = mensaje.lower().split()

    p_spam_mensaje = p_spam
    p_ham_mensaje = p_ham

    for palabra in mensaje:
        if palabra in param_spam:
            p_spam_mensaje *= param_spam[palabra]

        if palabra in param_ham:
            p_ham_mensaje *= param_ham[palabra]

    print('P(Spam|mensaje):', p_spam_mensaje)
    print('P(Ham|mensaje):', p_ham_mensaje)

    if p_ham_mensaje > p_spam_mensaje:
        print('Label: Ham')
    elif p_ham_mensaje < p_spam_mensaje:
        print('Label: Spam')
    else:
        print('Igual de probable, un humano debe decidir!')

def clasifica_test(mensaje):
    mensaje = re.sub('\W', ' ', mensaje)
    mensaje = mensaje.lower().split()

    p_spam_mensaje = p_spam
    p_ham_mensaje = p_ham

    for palabra in mensaje:
        if palabra in param_spam:
            p_spam_mensaje *= param_spam[palabra]
        if palabra in param_ham:
            p_ham_mensaje *= param_ham[palabra]

    if p_ham_mensaje > p_spam_mensaje:
        return 'ham'
    elif p_ham_mensaje < p_spam_mensaje:
        return 'spam'
    else:
        return 'Igual de probable'

''' ENTREGA 9:
 Completa la función 
clasifica_test()
 y usándola con los mensajes de test calcula el 
accuracy. Recuerda que 
accuracy = mensajes bien clasificados / total de mensajes.
'''
test['prediccion'] = test['SMS'].apply(clasifica_test)
print("== Test con predicciones realizadas:\n", test.head())

correctas = 0
total = test.shape[0]
for indice, fila in test.iterrows():
    if fila['Label'] == fila['prediccion']:
        correctas += 1

print(f'== Correctas {correctas} de {total} Accuracy: {correctas/total:.4f}')


