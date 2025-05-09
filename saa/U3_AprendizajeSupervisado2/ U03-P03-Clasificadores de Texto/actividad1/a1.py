# -*- coding: utf-8 -*-
train_spam = ['send us your password', 'review our website',
              'send your password', 'send us your account']
train_ham = ['Your activity report', 'benefits physical activity',
             'the importance vows']
test_emails = {'spam': ['renew your password', 'renew your vows'],
               'ham': ['benefits of our account', 'the importance of physical activity']}



# Hacer un vocabulario de palabras únicas que aparecen en los mails spam
vocab_palabras_spam = []
for frase in train_spam:
    frase_como_lista = frase.split()
    for w in frase_como_lista:
        vocab_palabras_spam.append(w)
print(vocab_palabras_spam)

vocab_palabras_spam_unicas = list(dict.fromkeys(vocab_palabras_spam))
print(vocab_palabras_spam_unicas)


# Probabilidades de cada palabra condicionado a spam
dict_spamicidad = {}
for w in vocab_palabras_spam_unicas:
    mails_con_w = 0  # contador
    for frase in train_spam:
        if w in frase:
            mails_con_w += 1
    print(f"Número de mails spam con la palabra '{w}': {mails_con_w}")
    spamicidad = (mails_con_w + 1) / (len(train_spam) + 2) # suavizado
    print(f"Spamicidad de la palabra '{w}': {spamicidad}\n")
    dict_spamicidad[w.lower()] = spamicidad

# Hacer un vocabulario de palabras únicas que aparecen en los mails ham
vocab_palabras_ham = []
for frase in train_ham:
    frase_como_lista = frase.lower().split()
    for w in frase_como_lista:
        vocab_palabras_ham.append(w)

vocab_palabras_ham_unicas = list(dict.fromkeys(vocab_palabras_ham))

# Probabilidades de cada palabra condicionado a ham
dict_hamicidad = {}
for w in vocab_palabras_ham_unicas:
    mails_con_w = 0  # contador
    for frase in train_ham:
        if w in frase.lower():
            mails_con_w += 1
    hamicidad = (mails_con_w + 1) / (len(train_ham) + 2)  # suavizado
    dict_hamicidad[w] = round(hamicidad, 1)

print("Hamicidad:", dict_hamicidad)


# Probabilidades de spam y ham
prob_spam = len(train_spam) / (len(train_spam) + len(train_ham))
print("P(S)=", prob_spam)
prob_ham = len(train_ham) / (len(train_spam) + len(train_ham))
print("P(H)=", prob_ham)


# Dividir los mail en palabras únicas
distintas_palabras_como_frase_test = []
for frase in test_emails['spam'] + test_emails['ham']:
    frase_como_lista = frase.split()
    sentencia = []
    for w in frase_como_lista:
        sentencia.append(w)
    distintas_palabras_como_frase_test.append(sentencia)
print(distintas_palabras_como_frase_test)

test_spam_tokenizado = distintas_palabras_como_frase_test[0:2]
test_ham_tokenizado = distintas_palabras_como_frase_test[2:4]

print("Test spam tokenizado", test_spam_tokenizado)
print("Test ham tokenizado", test_ham_tokenizado)


# Eliminar palabras de test sin datos en los datos de train
spam_test_reducido = []
for frase in test_spam_tokenizado:
    palabras_filtradas = []
    for w in frase:
        if w in vocab_palabras_spam_unicas:
            print(f"'{w}', ok (en vocabulario spam)")
            palabras_filtradas.append(w)
        elif w in vocab_palabras_ham_unicas:
            print(f"'{w}', ok (en vocabulario ham)")
            palabras_filtradas.append(w)
        else:
            print(f"'{w}', sin información en train")
    spam_test_reducido.append(palabras_filtradas)
print("Test de spam reducido:", spam_test_reducido)

# Eliminar palabras de test sin datos en los datos de train
ham_test_reducido = []
for frase in test_ham_tokenizado:
    palabras_filtradas = []
    for w in frase:
        if w in vocab_palabras_ham_unicas:
            print(f"'{w}', ok (en vocabulario ham)")
            palabras_filtradas.append(w)
        elif w in vocab_palabras_spam_unicas:
            print(f"'{w}', ok (en vocabulario spam)")
            palabras_filtradas.append(w)
        else:
            print(f"'{w}', sin información en train")
    ham_test_reducido.append(palabras_filtradas)

print("Test de ham reducido:", ham_test_reducido)

# stemmed
test_spam_stemmed = []
poco_importantes = ['us', 'the', 'of', 'your'] # palabras no clave
for email in spam_test_reducido:
    email_limpiado = []
    for w in email:
        if w in poco_importantes:
            print(f"Eliminar '{w}'")
        else:
            email_limpiado.append(w)
    test_spam_stemmed.append(email_limpiado)
print("Test spam stemmed:", test_spam_stemmed)

# stemmed para test ham
test_ham_stemmed = []
for email in ham_test_reducido:
    email_limpiado = []
    for w in email:
        if w in poco_importantes:
            print(f"Eliminar '{w}'")
        else:
            email_limpiado.append(w)
    test_ham_stemmed.append(email_limpiado)

print("Test ham stemmed:", test_ham_stemmed)

def multiplica(lista): # multiplica las probs de las palabras de la lista
    total_prob = 1
    for i in lista:
        total_prob *= i
    return total_prob

def Bayes(email):
    probs = []
    for w in email: # para cada palabra w del mail
        PS = prob_spam
        print(f"P(S)=", PS)
        try:
            P_ws = dict_spamicidad[w]
            print(f"P('{w}'|spam)=", P_ws)
        except KeyError:
            P_ws = 1 / (len(train_spam) + 2) # Aplicar suavizado a palabras no vistas en spam
            print(f"P('{w}'|spam)=", P_ws)

        PH = prob_ham
        print(f"P(H)=", PH)
        try:
            P_wh = dict_hamicidad[w]
            print(f"P('{w}'|ham)=", P_wh)
        except KeyError:
            P_wh = 1 / (len(train_ham) + 2) # Aplicar smoothing
            print(f"P('{w}'|ham)=", P_wh)

        P_spam_BAYES = (P_ws * PS) / ((P_ws * PS) + (P_wh * PH))
        print(f"Usando Bayes P(spam|'{w}')=", P_spam_BAYES)
        probs.append(P_spam_BAYES)
    print(f"Probabilidades de todas las palabras del mail: {probs}")
    clasificacion = multiplica(probs)
    if clasificacion >= 0.5:
        print(f"El email es SPAM: P(spam)={clasificacion * 100:.4f}%")
    else:
        print(f"El email es HAM: P(spam)={clasificacion * 100:.4f}%")
    return clasificacion

for email in test_spam_stemmed:
    print(f"===== Test mail SPAM {email} =====")
    Bayes(email)

for email in test_ham_stemmed:
    print(f"===== Test mail HAM {email} =====")
    Bayes(email)

