import pandas as pd

dataset = pd.read_csv('../Datos2.csv', encoding='utf-8')

# obtenemos las dos primeras filas
print(f'metodo head {dataset.head(2)}')
# obtenemos las dos ultimas filas
print(f'metodo tail{dataset.tail(2)}')

# descripcion de las columnas
print(f'describe {dataset.describe()}')
# info de las columnas
print(f'info {dataset.info()}')

# columnas
print(f'Columnas {dataset.columns}')

# localizar datos con iloc
print(f'iloc {dataset.iloc[0:1]}')
# traspuesta
print(f'Traspuesta \n {dataset.T}')

# filtrar valores
print(f'filtrado por peso {dataset[dataset['Peso'] > 72]}')

# borrar columna 1 = columna, 0 = fila

print(f'borrado de peso {dataset.drop(['Peso'], axis=1)}')
print(dataset)