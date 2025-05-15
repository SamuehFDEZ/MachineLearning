import csv
import os
from objeto_email import Email

LABELS_PATH = "../../U03_Recursos/U03_P03_Texto/SPAMTrain.label"
TRAINING_DIR = "../../U03_Recursos/U03_P03_Texto/training"
OUTPUT_CSV = "./dataset.csv"

def genera_dataset():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        lineas = f.readlines()

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline='') as salida:
        escritor = csv.writer(salida)
        escritor.writerow(["label", "texto"])  # Cabecera del CSV

        for linea in lineas:
            try:
                if not linea.strip():
                    continue  # Saltar líneas vacías

                partes = linea.strip().split()
                if len(partes) != 2:
                    print(f"Línea mal formateada: {linea.strip()}")
                    continue

                label, nombre_fichero = partes
                ruta_fichero = os.path.join(TRAINING_DIR, nombre_fichero)

                with open(ruta_fichero, "rb") as archivo_email:
                    mail = Email(archivo_email)

                    subject = mail.subject() or ""
                    body = mail.body() or ""

                    texto_completo = f"{subject} {body}".replace('\n', ' ').replace('\r', ' ')
                    escritor.writerow([label, texto_completo.strip()])
            except Exception as e:
                print(f"Error procesando {linea.strip()}: {e}")


if __name__ == "__main__":
    genera_dataset()
