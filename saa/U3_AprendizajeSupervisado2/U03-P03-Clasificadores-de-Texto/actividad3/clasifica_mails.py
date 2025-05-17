# entrena_modelos.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction import text
import joblib

# Cargar dataset
df = pd.read_csv('dataset.csv')
# Cargar dataset
df = df.dropna(subset=['texto'])  # Elimina cualquier fila con texto nulo

X = df['texto']
y = df['label']


# Separar en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Stopwords en inglés + español ampliado (convertido a lista)
stop_words = list(text.ENGLISH_STOP_WORDS.union({
    'de', 'la', 'que', 'el', 'y', 'en', 'los', 'del', 'se', 'las', 'por',
    'un', 'para', 'con', 'una', 'no', 'su', 'al', 'lo', 'como', 'más'
}))


# Diccionario de modelos
modelos = {
    'NaiveBayes': MultinomialNB(),
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'SVM': LinearSVC(),
    'RandomForest': RandomForestClassifier(n_estimators=100),
    'DecisionTree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier()
}

mejor_score = 0
mejor_modelo = None
nombre_mejor_modelo = ""

# Entrenamiento y evaluación
for nombre, modelo in modelos.items():
    print(f"\n--- Entrenando {nombre} ---")
    pipe = make_pipeline(TfidfVectorizer(stop_words=stop_words), modelo)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    reporte = classification_report(y_test, y_pred)
    print(reporte)

    # Guardar cada modelo individual
    joblib.dump(pipe, f'modelo_{nombre}.pkl')

    # Guardar el mejor modelo según accuracy
    score = pipe.score(X_test, y_test)
    if score > mejor_score:
        mejor_score = score
        mejor_modelo = pipe
        nombre_mejor_modelo = nombre

# Guardar el mejor modelo global
print(f"\nMejor modelo: {nombre_mejor_modelo} con accuracy {mejor_score:.2f}")
joblib.dump(mejor_modelo, 'mejor_modelo.pkl')
