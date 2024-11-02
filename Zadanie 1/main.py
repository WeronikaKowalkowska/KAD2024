import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dane = pd.read_csv('dane.csv',
                   header = None,
                   names = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Species'])


# Liczności poszczególnych gatunków

gatunki_liczebność = (dane['Species']).value_counts()
gatunki_liczebność.index = ['Setosa', 'Versicolor', 'Virginica']
gatunki_liczebność.loc['Razem'] = gatunki_liczebność.sum()
gatunki_liczebność.name = 'Liczebność'

liczności = pd.DataFrame()
liczności['Liczebność'] = gatunki_liczebność
liczności['Udział Procentowy (%)'] = gatunki_liczebność / gatunki_liczebność[0:3].sum() * 100
liczności.index.name = 'Gatunek'

pd.set_option('display.float_format', '{:.1f}'.format)
print('Liczności poszczególnych gatunków:')
print(liczności.to_string(index=True, header=True, justify='left'))


# Miary rozkładu każdej cechy (miary położenia rozkładu)

charakterystyka = pd.DataFrame(columns=['Minimum', 'Śr. arytm.', '± Odch. stand.', 'Q1', 'Q3', 'Mediana', 'Maksimum'])

charakterystyka.loc["Długość działki kielicha (cm)", "Minimum"] = (dane['Sepal length']).min()
charakterystyka.loc["Długość działki kielicha (cm)", "Śr. arytm."] = (dane['Sepal length']).mean()
charakterystyka.loc["Długość działki kielicha (cm)", "± Odch. stand."] = (dane['Sepal length']).std()
charakterystyka.loc["Długość działki kielicha (cm)", "Q1"] = (dane['Sepal length']).quantile(0.25)
charakterystyka.loc["Długość działki kielicha (cm)", "Q3"] = (dane['Sepal length']).quantile(0.75)
charakterystyka.loc["Długość działki kielicha (cm)", "Mediana"] = (dane['Sepal length']).median()
charakterystyka.loc["Długość działki kielicha (cm)", "Maksimum"] = (dane['Sepal length']).max()

charakterystyka.loc["Szerokość działki kielicha (cm)", "Minimum"] = (dane['Sepal width']).min()
charakterystyka.loc["Szerokość działki kielicha (cm)", "Śr. arytm."] = (dane['Sepal width']).mean()
charakterystyka.loc["Szerokość działki kielicha (cm)", "± Odch. stand."] = (dane['Sepal width']).std()
charakterystyka.loc["Szerokość działki kielicha (cm)", "Q1"] = (dane['Sepal width']).quantile(0.25)
charakterystyka.loc["Szerokość działki kielicha (cm)", "Q3"] = (dane['Sepal width']).quantile(0.75)
charakterystyka.loc["Szerokość działki kielicha (cm)", "Mediana"] = (dane['Sepal width']).median()
charakterystyka.loc["Szerokość działki kielicha (cm)", "Maksimum"] = (dane['Sepal width']).max()

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



