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
charakterystyka.loc["Długość działki kielicha (cm)", "± Odch. stand."] = (dane['Sepal length']).std()
charakterystyka.loc["Długość działki kielicha (cm)", "Q1"] = (dane['Sepal length']).quantile(0.25)
charakterystyka.loc["Długość działki kielicha (cm)", "Q3"] = (dane['Sepal length']).quantile(0.75)
charakterystyka.loc["Długość działki kielicha (cm)", "Mediana"] = np.median(dlKielicha)
charakterystyka.loc["Długość działki kielicha (cm)", "Maksimum"] = dlKielicha.max()

charakterystyka.loc["Szerokość działki kielicha (cm)", "Minimum"] = szerKielicha.min()
charakterystyka.loc["Szerokość działki kielicha (cm)", "Śr. arytm."] = szerKielicha.mean()
charakterystyka.loc["Szerokość działki kielicha (cm)", "± Odch. stand."] = (dane['Sepal width']).std()
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


# Współczynnik korelacji liniowej Pearsona (szerokość i długość działki kielicha)
wPSZkDk = pd.DataFrame (columns = ['Długość działki kielicha (cm)', 'Szerokość działki kielicha (cm)'])
wPSZkDk['Długość działki kielicha (cm)'] = dlKielicha
wPSZkDk['Szerokość działki kielicha (cm)'] = szerKielicha
wPSZkDk_value = wPSZkDk.corr('pearson').loc['Długość działki kielicha (cm)', 'Szerokość działki kielicha (cm)']

# Współczynnik korelacji liniowej Pearsona (szerokość płatka i długość działki kielicha)
wPSZpDk = pd.DataFrame (columns = ['Długość działki kielicha (cm)', 'Szerokość płatka (cm)'])
wPSZpDk['Długość działki kielicha (cm)'] = dlKielicha
wPSZpDk['Szerokość płatka (cm)'] = szerPlatka
wPSZpDk_value = wPSZpDk.corr('pearson').loc['Długość działki kielicha (cm)', 'Szerokość płatka (cm)']

# Współczynnik korelacji liniowej Pearsona (szerokość płatka i szerokość działki kielicha)
wPSZpSZk = pd.DataFrame (columns = ['Szerokość działki kielicha (cm)', 'Szerokość płatka (cm)'])
wPSZpSZk['Szerokość działki kielicha (cm)'] = szerKielicha
wPSZpSZk['Szerokość płatka (cm)'] = szerPlatka
wPSZpSZk_value = wPSZpSZk.corr('pearson').loc['Szerokość działki kielicha (cm)', 'Szerokość płatka (cm)']

# Współczynnik korelacji liniowej Pearsona (długość płatka i długość działki kielicha)
wPDpDk = pd.DataFrame (columns = ['Długość działki kielicha (cm)', 'Długość płatka (cm)'])
wPDpDk['Długość działki kielicha (cm)'] = dlKielicha
wPDpDk['Długość płatka (cm)'] = dlPlatka
wPDpDk_value = wPDpDk.corr('pearson').loc['Długość działki kielicha (cm)', 'Długość płatka (cm)']

# Współczynnik korelacji liniowej Pearsona (długość płatka i szerokość działki kielicha)
wPDpSZk = pd.DataFrame (columns = ['Szerokość działki kielicha (cm)', 'Długość płatka (cm)'])
wPDpSZk['Szerokość działki kielicha (cm)'] = szerKielicha
wPDpSZk['Długość płatka (cm)'] = dlPlatka
wPDpSZk_value= wPDpSZk.corr('pearson').loc['Szerokość działki kielicha (cm)', 'Długość płatka (cm)']

# Współczynnik korelacji liniowej Pearsona (długość płatka i szerokość płatka)
wPDpSZp = pd.DataFrame (columns = ['Długość płatka (cm)', 'Szerokość płatka (cm)'])
wPDpSZp['Szerokość płatka (cm)'] = szerPlatka
wPDpSZp['Długość płatka (cm)'] = dlPlatka
wPDpSZp_value= wPDpSZp.corr('pearson').loc['Długość płatka (cm)', 'Szerokość płatka (cm)']



# Wykres punktowy (szerokość i długość działki kielicha)
plt.scatter(wPSZkDk['Długość działki kielicha (cm)'], wPSZkDk['Szerokość działki kielicha (cm)'], marker='o')
plt.title("r = " + str(wPSZkDk_value.round(2)) + "; y =")
plt.xlabel('Długość działki kielicha (cm)')
plt.ylabel('Szerokość działki kielicha (cm)')
# plt.xlim(4, 8)
# plt.xticks(np.arange(4, 9, 1))
plt.show()

# Wykres punktowy (szerokość płatka i długość działki kielicha)
plt.scatter(wPSZpDk['Długość działki kielicha (cm)'], wPSZpDk['Szerokość płatka (cm)'], marker='o')
plt.title("r = " + str(wPSZpDk_value.round(2)) + "; y =")
plt.xlabel('Długość działki kielicha (cm)')
plt.ylabel('Szerokość płatka (cm)')
plt.show()

# Wykres punktowy (szerokość płatka i szerokość działki kielicha)
plt.scatter(wPSZpSZk['Szerokość działki kielicha (cm)'], wPSZpSZk['Szerokość płatka (cm)'], marker='o')
plt.title("r = " + str(wPSZpSZk_value.round(2)) + "; y =")
plt.xlabel('Szerokość działki kielicha (cm)')
plt.ylabel('Szerokość płatka (cm)')
plt.show()