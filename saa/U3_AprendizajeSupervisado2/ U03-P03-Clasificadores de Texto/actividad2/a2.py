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