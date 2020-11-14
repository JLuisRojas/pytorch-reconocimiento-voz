import pandas as pd
import numpy as np

def main():
    df = pd.read_csv("./dataset/cv-corpus-5.1-2020-06-22/es/train.tsv", sep='\t')
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

    for i in range(1, 51):
        palabra = keys[-i]
        num = vocabulario[palabra]
        if len(palabra) >= 4:
            print(f"{palabra}: {num}")



if __name__ == "__main__":
    main()