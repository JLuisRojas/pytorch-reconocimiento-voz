import pandas as pd
import numpy as np

def main():
    df = pd.read_csv("./dataset/cv-corpus-5.1-2020-06-22/es/test.tsv", sep='\t')
    vocabulario = {}
    #palabra = "encuentra" #1070
    audios = []
    cadenas = []
    for indice, renglon in df.iterrows():
        guardar = False
        cadena = renglon["sentence"]
        for palabra in cadena.split():
            if palabra in vocabulario:
                vocabulario[palabra] += 1
            else:
                vocabulario[palabra] = 1

            if palabra == "encuentra":
                guardar = True

    #if guardar:
    #	audios.append(renglon["path"])
    #	cadenas.append(cadena)
        

    vocabulario = {k: v for k,v in sorted(vocabulario.items(), key=lambda item: item[1])}
    print(vocabulario)
    print(len(df.index))

    keys = list(vocabulario)

    for i in range(1, 101):
        palabra = keys[-i]
        num = vocabulario[palabra]
        if len(palabra) >= 5:
            pass
            #print(f"{palabra}: {num}")

    palabras = ["encuentra", "parte", "tiene", "fueron", "entre", "sobre", "ciudad", "durante", "nombre", "embargo", "puede", "primer", "Actualmente", "hasta", "mismo", "Universidad"]

    for p in palabras:
        print(f"{p}: {vocabulario[p]}")


if __name__ == "__main__":
    main()