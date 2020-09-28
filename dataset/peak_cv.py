"""
Script que obtiene un cantidad pequena de datos
para relaizar pruebas para entrenamiento en
common voice
"""

import pandas as pd
import shutil

num_ejemplos = 50
distrib = 'dev'

ruta_original = './cv-corpus-5.1-2020-06-22/es/'
ruta_copia = './common-voice/es/'
ruta_clips = 'clips/'

def main():
    # leer tabla de datos
    df = pd.read_csv(f"{ruta_original}{distrib}.tsv", sep='\t')

    # obtener solo n ejemplos
    df = df.head(num_ejemplos)

    # transferirlos a otra carpeta
    for idx, row in df.iterrows():
        a = row['path']
        audio = f"{ruta_original}{ruta_clips}{a}"
        shutil.copyfile(audio, f"{ruta_copia}{ruta_clips}{a}")

    # guardar subtabla de datos
    df.to_csv(f"{ruta_copia}{distrib}.tsv", sep='\t')

if __name__ == "__main__":
    main()

