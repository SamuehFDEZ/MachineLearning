import os
from PIL import Image

# Ruta de la carpeta con las imágenes .jpg
carpeta = 'C:/Users/ESP/Desktop/animals'  # Cambia esto por la ruta real
contador = 1

# Crear carpeta de salida si se desea
# carpeta_salida = os.path.join(carpeta, "convertidas")
# os.makedirs(carpeta_salida, exist_ok=True)

for archivo in os.listdir(carpeta):
    if archivo.lower().endswith(".png"):
        ruta_jpg = os.path.join(carpeta, archivo)
        nuevo_nombre = f"peligro-samartlop-{contador}.png"
        ruta_png = os.path.join(carpeta, nuevo_nombre)

        # Opcional: eliminar el archivo original

        contador += 1

print(contador)
print("Conversión y renombrado completados.")