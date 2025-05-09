# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar datos
spam_df = pd.read_csv('../../U03_Recursos/U03_P03_Texto/emails.csv')
print(spam_df.head(10))
print(spam_df.info())

# Visualizar datos
ham = spam_df[spam_df['spam'] == 0]
spam = spam_df[spam_df['spam'] == 1]
print(f'Porcentaje de Spam = {len(spam)/len(spam_df) * 100:.4f}%')
print(f'Porcentaje de Ham = {len(ham)/len(spam_df)*100:.4f}%')
sns.countplot(x='spam', data=spam_df, label='Spam vs Ham')
plt.show()

''' ENTREGA 5:
 ¿Cuántos emails contiene el dataset? ¿Qué porcentaje de ellos son spam? 
5728 y 23.8827%
'''

# Aplicar la vectorización automática
from sklearn.feature_extraction.text import CountVectorizer
c_v = CountVectorizer()
spam_h_c_v = c_v.fit_transform( spam_df['text'] )
print("CountVectorizer: Nombres de caracteristicas:\n", c_v.get_feature_names_out())
print("CountVectorizer: diseño", spam_h_c_v.shape)
label = spam_df['spam']
X = spam_h_c_v
y = label
print("Dimensiones de X", X.shape)

''' ENTREGA 6:
 El problema de no eliminar palabras comunes es que la cantidad de palabras o tokens que debes 
procesar se dispara mucho y cada una de ellas se considera una característica de los datos de entrenamiento 
¿Cuántas distintas palabras ha encontrado el vectorizador? (5728, 37303) '''

# Dividir en train y test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# Crear el modelo naive Bayes de tipo Multinomial
from sklearn.naive_bayes import MultinomialNB
nbc = MultinomialNB()
nbc.fit(X_train, y_train)

# Imprimir resultados de validación: mapa de calor y matriz de confusión.
from sklearn.metrics import classification_report, confusion_matrix
y_pred_train = nbc.predict(X_train)
cm = confusion_matrix(y_train, y_pred_train)
sns.heatmap(cm, annot=True)
plt.show()
y_pred_test = nbc.predict(X_test)
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True)
print(classification_report(y_test, y_pred_test))
plt.show()

'''ENTREGA 7:
 En el problema de clasificar spam que es peor, ¿Tener un falso positivo (decir que un mail es 
spam cuando no lo es) o un falso negativo (decir que un mail es ham cuando en realidad es spam)? 

Un falso positivo puede hacer que pierdas correos importantes (ham clasificado como spam).

Un falso negativo solo implica que un spam aparece en la bandeja de entrada, lo cual es molesto pero menos crítico.

• Muestra la matriz de confusión del entrenamiento y del test.
 • ¿Parece que generaliza bien este modelo? (ar
 gumenta la respuesta)
 El modelo muestra un 99% de accuracy en test, y tanto precision como recall 
 son muy altos (>= 0.97). Además, no hay sobreajuste evidente: los resultados del entrenamiento y del test son similares.
 
 • Para
 la del test indica donde comete más fallos cuantitativamente (5 errores, 4 errores, …) y 
porcentualmente (fallo 5 veces en decir que es spam de 100 emails spam que tengo, es el 5% de error).
 • Pasa
 captura del informe de la clasificación, e indica si los valores de las métricas darían por bueno el 
modelo según lo que hayas respondido en la primera cuestión

 Dado que los falsos positivos (aunque más problemáticos) son solo un 1% del total de hams, el modelo es excelente y aceptable en un entorno real.
'''



