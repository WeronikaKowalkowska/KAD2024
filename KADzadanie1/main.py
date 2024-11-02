import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# Read the CSV file into a DataFrame
header=['dlKielicha','szerKielicha', 'dlPlatka', 'szerPlatka', 'gatunek']
df = pd.read_csv('./Zadanie 1 Dane-20241102/data1.csv')
#tworze dodatkowy plik aby dodac naglowki do kolumn i wczytac kolumny jako kolejki
df.to_csv("dane_z_naglowkiem.csv")
df2=pd.read_csv("dane_z_naglowkiem.csv" ,names=header, dtype={'dlKielicha':np.float32,'szerKielicha':np.float32,'dlPlatka':np.float32,'szerPlatka':np.float32,'gatunek':np.int32} )

dlKielicha = np.array(df2['dlKielicha'])
szerKielicha = np.array(df2['szerKielicha'])
dlPlatka = np.array(df2['dlPlatka'])
szerPlatka = np.array(df2['szerPlatka'])
gatunek = np.array(df2['gatunek'])

liczebnoscSet=0
liczebnoscVer=0
liczebnoscVir=0

for i in range(0,len(gatunek)):
    if gatunek[i] == 0:
        liczebnoscSet += 1
    elif gatunek[i] == 1:
        liczebnoscVer += 1
    else:
        liczebnoscVir += 1

liczebnoscCala=liczebnoscSet+liczebnoscVer+liczebnoscVir
col_names=["gatunek","liczebnosc", "%"]
table_data=[["setosa",liczebnoscSet, liczebnoscSet/liczebnoscCala*100],
            ["versicolor",liczebnoscVer, liczebnoscVer/liczebnoscCala*100],
            ["virginica",liczebnoscVir, liczebnoscVir/liczebnoscCala*100],
            ["razem",liczebnoscCala,100]]
print(tabulate(table_data,headers=col_names))


