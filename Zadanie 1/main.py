import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dane = pd.read_csv('dane.csv',
                   header = None,
                   names = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Species'])

dlKielicha = np.array(dane['Sepal length'])
szerKielicha = np.array(dane['Sepal width'])
dlPlatka = np.array(dane['Petal length'])
szerPlatka = np.array(dane['Petal width'])
gatunek = np.array(dane['Species'])

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


# Liczności poszczególnych gatunków


gatunki_liczebność = pd.DataFrame(columns=['Liczebność'])
gatunki_liczebność.loc['Setosa'] = liczebnoscSet
gatunki_liczebność.loc['Versicolor'] = liczebnoscVer
gatunki_liczebność.loc['Virginica'] =liczebnoscVir
gatunki_liczebność.loc['Razem'] = liczebnoscCala

liczności = pd.DataFrame()
liczności['Liczebność'] = gatunki_liczebność
liczności['Udział Procentowy (%)'] = gatunki_liczebność / gatunki_liczebność[0:3].sum() * 100
liczności.index.name = 'Gatunek'

pd.set_option('display.float_format', '{:.1f}'.format)
print('Liczności poszczególnych gatunków:')
print(liczności.to_string(index=True, header=True, justify='left'))
print(" ")

# Miary rozkładu każdej cechy (miary położenia rozkładu)

charakterystyka = pd.DataFrame(columns=['Minimum', 'Śr. arytm.', '± Odch. stand.', 'Q1', 'Q3', 'Mediana', 'Maksimum'])

charakterystyka.loc["Długość działki kielicha (cm)", "Minimum"] = dlKielicha.min()
charakterystyka.loc["Długość działki kielicha (cm)", "Śr. arytm."] = dlKielicha.mean()
charakterystyka.loc["Długość działki kielicha (cm)", "± Odch. stand."] = dlKielicha.std()
charakterystyka.loc["Długość działki kielicha (cm)", "Q1"] = (dane['Sepal length']).quantile(0.25)
charakterystyka.loc["Długość działki kielicha (cm)", "Q3"] = (dane['Sepal length']).quantile(0.75)
charakterystyka.loc["Długość działki kielicha (cm)", "Mediana"] = np.median(dlKielicha)
charakterystyka.loc["Długość działki kielicha (cm)", "Maksimum"] = dlKielicha.max()

charakterystyka.loc["Szerokość działki kielicha (cm)", "Minimum"] = szerKielicha.min()
charakterystyka.loc["Szerokość działki kielicha (cm)", "Śr. arytm."] = szerKielicha.mean()
charakterystyka.loc["Szerokość działki kielicha (cm)", "± Odch. stand."] = szerKielicha.std()
charakterystyka.loc["Szerokość działki kielicha (cm)", "Q1"] = (dane['Sepal width']).quantile(0.25)
charakterystyka.loc["Szerokość działki kielicha (cm)", "Q3"] = (dane['Sepal width']).quantile(0.75)
charakterystyka.loc["Szerokość działki kielicha (cm)", "Mediana"] = np.median(szerKielicha)
charakterystyka.loc["Szerokość działki kielicha (cm)", "Maksimum"] = szerKielicha.max()

pd.set_option('display.float_format', '{:.2f}'.format)
print('Charakterystyka cech irysów:')

print(charakterystyka.to_string(index=True, header=True, justify='left'))

# Dane posegregowane według gatunków
daneSetosa = dane[dane['Species']==0]
daneVersicolor = dane[dane['Species']==1]
daneVirginica = dane[dane['Species']==2]

# Histogram długości działki kielicha
plt.hist(dane['Sepal length'], bins=8, edgecolor='black', linewidth=1.5)
plt.title('Długość działki kielicha')
plt.xlabel('Długość (cm)')
plt.ylabel('Liczebność')
plt.show()

# Wykres pudełkowy długości działki kielicha
sepalLengthByType = [daneSetosa['Sepal length'], daneVersicolor['Sepal length'], daneVirginica['Sepal length']]

fig,ax = plt.subplots()
ax.boxplot(sepalLengthByType, tick_labels=['setosa', 'versicolor', 'virginica'])
plt.title('Długość działki kielicha')
ax.set_xlabel('Gatunek')
ax.set_ylabel('Długość (cm)')
plt.show()

# Histogram szerokości działki kielicha
plt.hist(dane['Sepal width'], bins=8, edgecolor='black', linewidth=1.5)
plt.title('Szerokość działki kielicha')
plt.xlabel('Szerokość (cm)')
plt.ylabel('Liczebność')
plt.show()

# Wykres pudełkowy szerokości działki kielicha
sepalWidthByType = [daneSetosa['Sepal width'], daneVersicolor['Sepal width'], daneVirginica['Sepal width']]

fig1,ax1 = plt.subplots()
ax1.boxplot(sepalWidthByType, tick_labels=['setosa', 'versicolor', 'virginica'])
plt.title('Szerokość działki kielicha')
ax1.set_xlabel('Gatunek')
ax1.set_ylabel('Szerokość (cm)')
plt.show()

# Histogram długości płatka


# Wykres pudełkowy długości płatka


# Histogram szerokości płatka


# Wykres pudełkowy szerokości płatka



