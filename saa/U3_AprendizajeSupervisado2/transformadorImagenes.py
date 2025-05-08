from PIL import Image
import os

# Ruta de entrada y salida
carpeta = r"C:\Users\ESP\Desktop\MachineLearning\saa\U3_AprendizajeSupervisado2\U03_Recursos\pintores"
carpeta_salida = os.path.join(carpeta, "convertidas_png")
os.makedirs(carpeta_salida, exist_ok=True)

for archivo in os.listdir(carpeta):
    nombre, ext = os.path.splitext(archivo)
    ruta = os.path.join(carpeta, archivo)

    try:
        with Image.open(ruta) as img:
            img = img.convert("RGB")  # Convertir a RGB para evitar problemas con transparencias
            nueva_ruta = os.path.join(carpeta_salida, f"{nombre}.png")
            img.save(nueva_ruta, "PNG")
            print(f"✅ Convertido: {archivo} → {nombre}.png")
    except Exception as e:
        print(f"❌ Error con {archivo}: {e}")
