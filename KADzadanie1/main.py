import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
header=['dlKielicha','szerKielicha', 'dlPlatka', 'szerPlatka', 'gatunek']
df = pd.read_csv('./Zadanie 1 Dane-20241102/data1.csv')
#tworze dodatkowy plik aby dodac naglowki do kolumn i wczytac kolumny jako kolejki 
df.to_csv("dane_z_naglowkiem.csv",header=header,index=False)
df2=pd.read_csv("dane_z_naglowkiem.csv" ,dtype={'dlKielicha':np.float32,'szerKielicha':np.float32,'dlPlatka':np.float32,'szerPlatka':np.float32,'gatunek':np.int32} )

dlKielicha = np.array(df2['dlKielicha'])
szerKielicha = np.array(df2['szerKielicha'])
dlPlatka = np.array(df2['dlPlatka'])
szerPlatka = np.array(df2['szerPlatka'])
gatunek = np.array(df2['gatunek'])
