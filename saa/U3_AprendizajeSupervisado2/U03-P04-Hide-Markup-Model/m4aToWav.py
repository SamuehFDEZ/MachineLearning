from pydub import AudioSegment
from pydub.utils import which
import os

AudioSegment.converter = which("ffmpeg")  # Forzar pydub a usar ffmpeg

def convertir_m4a_a_wav(carpeta_entrada):
    for archivo in os.listdir(carpeta_entrada):
        if archivo.lower().endswith(".m4a"):
            ruta_entrada = os.path.join(carpeta_entrada, archivo)
            nombre_base = os.path.splitext(archivo)[0]
            ruta_salida = os.path.join(carpeta_entrada, nombre_base + ".wav")

            try:
                sonido = AudioSegment.from_file(ruta_entrada, format="m4a")
                sonido.export(ruta_salida, format="wav")
                print(f"Convertido: {archivo} → {nombre_base}.wav")
            except Exception as e:
                print(f"Error con {archivo}: {e}")

# Ejecutar
carpeta_entrada = r"C:\Users\ESP\Desktop\MachineLearning\saa\U3_AprendizajeSupervisado2\U03_Recursos\U03_P04_Audios"
convertir_m4a_a_wav(carpeta_entrada)
