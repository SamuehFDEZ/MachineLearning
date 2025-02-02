import pandas as pd

# numero de columnas
columnas = ['jugados','ganados','perdidos']
filas = ['VCF','Betis','ATM','FCB']
datos = [{'jugados':3,'ganados':3,'perdidos':0},
         {'jugados':3,'ganados':2,'perdidos':1},
         {'jugados':3,'ganados':0,'perdidos':3},
         {'jugados':1,'ganados':2,'perdidos':2}]
df1 = pd.DataFrame(data=datos, index=filas, columns=columnas)

# acceder a todos los datos por el indice (mediante el nombre)
print(df1.loc['VCF'])

df1['empatados'] = df1['jugados'] -df1['ganados'] - df1['perdidos']
# acceder por posiciones
print(df1.iloc[0])

