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

charakterystyka.loc["Długość płatka (cm)", "Minimum"] = dlPlatka.min()
charakterystyka.loc["Długość płatka (cm)", "Śr. arytm."] = dlPlatka.mean()
charakterystyka.loc["Długość płatka (cm)", "± Odch. stand."] = (dane['Petal length']).std()
charakterystyka.loc["Długość płatka (cm)", "Q1"] = (dane['Petal length']).quantile(0.25)
charakterystyka.loc["Długość płatka (cm)", "Q3"] = (dane['Petal length']).quantile(0.75)
charakterystyka.loc["Długość płatka (cm)", "Mediana"] = np.median(dlPlatka)
charakterystyka.loc["Długość płatka (cm)", "Maksimum"] = dlPlatka.max()

charakterystyka.loc["Szerokość płatka (cm)", "Minimum"] = szerPlatka.min()
charakterystyka.loc["Szerokość płatka (cm)", "Śr. arytm."] = szerPlatka.mean()
charakterystyka.loc["Szerokość płatka (cm)", "± Odch. stand."] = (dane['Petal width']).std()
charakterystyka.loc["Szerokość płatka (cm)", "Q1"] = (dane['Petal width']).quantile(0.25)
charakterystyka.loc["Szerokość płatka (cm)", "Q3"] = (dane['Petal width']).quantile(0.75)
charakterystyka.loc["Szerokość płatka (cm)", "Mediana"] = np.median(szerPlatka)
charakterystyka.loc["Szerokość płatka (cm)", "Maksimum"] = szerPlatka.max()

pd.set_option('display.float_format', '{:.2f}'.format)
print('Charakterystyka cech irysów:')

print(charakterystyka.to_string(index=True, header=True, justify='left'))

# Dane posegregowane według gatunków
daneSetosa = dane[dane['Species']==0]
daneVersicolor = dane[dane['Species']==1]
daneVirginica = dane[dane['Species']==2]


# Histogram długości działki kielicha
bins = np.arange(4, 8.5, 0.5)
plt.hist(dane['Sepal length'], bins=bins, edgecolor='black', linewidth=1.5)
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
bins2 = np.arange(2, 4.6, 0.25)
plt.hist(dane['Sepal width'], bins=bins2, edgecolor='black', linewidth=1.5)
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
xAxisBins=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,4.5,5.0,5.5,6.0,6.5,7.0]
plt.hist(dane['Petal length'], bins=xAxisBins, edgecolor='black', linewidth=1.5)
plt.title('Długości płatka')
plt.xlabel('Długość (cm)')
plt.ylabel('Liczebność')
plt.show()

# Wykres pudełkowy długości płatka
petalLengthByType = [daneSetosa['Petal length'], daneVersicolor['Petal length'], daneVirginica['Petal length']]

fig2,ax2 = plt.subplots()
ax2.boxplot(petalLengthByType, tick_labels=['setosa', 'versicolor', 'virginica'])
plt.title('Długość płatka')
ax2.set_xlabel('Gatunek')
ax2.set_ylabel('Długość (cm)')
plt.show()


# Histogram szerokości płatka
xAxisBins=np.arange(0, 3, 0.25)
plt.hist(dane['Petal width'], bins=xAxisBins, edgecolor='black', linewidth=1.5)
plt.title('Szerokość płatka')
plt.xlabel('Szerokść (cm)')
plt.ylabel('Liczebność')
plt.show()

# Wykres pudełkowy szerokości płatka
petalWidthByType = [daneSetosa['Petal width'], daneVersicolor['Petal width'], daneVirginica['Petal width']]

fig3,ax3 = plt.subplots()
ax3.boxplot(petalWidthByType, tick_labels=['setosa', 'versicolor', 'virginica'])
plt.title('Szerokość płatka')
ax3.set_xlabel('Gatunek')
ax3.set_ylabel('Szerokość (cm)')
plt.show()

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

#a=cov(x,y)/var(x)
#b=sr y - a * sr x
"""covariance=np.cov(X, Y)
a=covariance[0,1]/np.var(X)
b=Y.mean()-a*X.mean()
line=np.array(a*X+b)"""

#1- Linia regresji (Y- szerokość i X- długość działki kielicha)
covariance=np.cov(dlKielicha, szerKielicha)
rLSZkDk_a=covariance[0,1]/np.var(dlKielicha)
rLSZkDk_b=szerKielicha.mean()-rLSZkDk_a*dlKielicha.mean()
rLSZkDk_line=np.array(rLSZkDk_a*dlKielicha+rLSZkDk_b)

#2- Linia regresji (szerokość płatka i X- długość działki kielicha)
covariance=np.cov(dlKielicha, szerPlatka)
rLSZpDk_a=covariance[0,1]/np.var(dlKielicha)
rLSZpDk_b=szerPlatka.mean()-rLSZpDk_a*dlKielicha.mean()
rLSZpDk_line=np.array(rLSZpDk_a*dlKielicha+rLSZpDk_b)

#3- Linia regresji (szerokość płatka i X- szerokość działki kielicha)
covariance=np.cov(szerKielicha, szerPlatka)
rLSZpSZk_a=covariance[0,1]/np.var(szerKielicha)
rLSZpSZk_b=szerPlatka.mean()-rLSZpSZk_a*szerKielicha.mean()
rLSZpSZk_line=np.array(rLSZpSZk_a*szerKielicha+rLSZpSZk_b)

#4- Linia regresji (długość płatka i X- długość działki kielicha)
covariance=np.cov(dlKielicha, dlPlatka)
rLDpDk_a=covariance[0,1]/np.var(dlKielicha)
rLDpDk_b=dlPlatka.mean()-rLDpDk_a*dlKielicha.mean()
rLDpDk_line=np.array(rLDpDk_a*dlKielicha+rLDpDk_b)

#5- Linia regresji (długość płatka i X- szerokość działki kielicha)
covariance=np.cov(szerKielicha, dlPlatka)
rLDpSZk_a=covariance[0,1]/np.var(szerKielicha)
rLDpSZk_b=dlPlatka.mean()-rLDpSZk_a*szerKielicha.mean()
rLDpSZk_line=np.array(rLDpSZk_a*szerKielicha+rLDpSZk_b)

#6- Linia regresji (X- długość płatka i szerokość płatka)
covariance=np.cov(dlPlatka, szerPlatka)
rLDpSZp_a=covariance[0,1]/np.var(dlPlatka)
rLDpSZp_b=szerPlatka.mean()-rLDpSZp_a*dlPlatka.mean()
rLDpSZp_line=np.array(rLDpSZp_a*dlPlatka+rLDpSZp_b)

# Wykres punktowy (szerokość i długość działki kielicha)
plt.scatter(wPSZkDk['Długość działki kielicha (cm)'], wPSZkDk['Szerokość działki kielicha (cm)'], marker='o')
plt.title("r = " + str(wPSZkDk_value.round(2)) + "; y =" + str(rLSZkDk_a.round(1)) + "x + " + str(rLSZkDk_b.round(1)))
plt.xlabel('Długość działki kielicha (cm)')
plt.ylabel('Szerokość działki kielicha (cm)')
# plt.xlim(4, 8)
# plt.xticks(np.arange(4, 9, 1))
plt.plot(wPSZkDk['Długość działki kielicha (cm)'], rLSZkDk_line, color = "r")
plt.show()

# Wykres punktowy (szerokość płatka i długość działki kielicha)
plt.scatter(wPSZpDk['Długość działki kielicha (cm)'], wPSZpDk['Szerokość płatka (cm)'], marker='o')
plt.title("r = " + str(wPSZpDk_value.round(2)) + "; y ="+ str(rLSZpDk_a.round(1)) + "x + " + str(rLSZpDk_b.round(1)))
plt.xlabel('Długość działki kielicha (cm)')
plt.ylabel('Szerokość płatka (cm)')
plt.plot(wPSZpDk['Długość działki kielicha (cm)'], rLSZpDk_line, color = "r")
plt.show()

# Wykres punktowy (szerokość płatka i szerokość działki kielicha)
plt.scatter(wPSZpSZk['Szerokość działki kielicha (cm)'], wPSZpSZk['Szerokość płatka (cm)'], marker='o')
plt.title("r = " + str(wPSZpSZk_value.round(2)) + "; y ="+ str(rLSZpSZk_a.round(1)) + "x - " + str(rLSZpSZk_b.round(1)))
plt.ylabel('Szerokość płatka (cm)')
plt.xlabel('Szerokość działki kielicha (cm)')
plt.plot(wPSZpSZk['Szerokość działki kielicha (cm)'], rLSZpSZk_line, color = "r")
plt.show()

# Wykres punktowy (długość płatka i długość działki kielicha)
plt.scatter(wPDpDk['Długość działki kielicha (cm)'], wPDpDk['Długość płatka (cm)'], marker='o')
plt.title("r = " + str(wPDpDk_value.round(2)) + "; y =" + str(rLDpDk_a.round(1))+ "x - " + str(rLDpDk_b.round(1)))
plt.xlabel('Długość działki kielicha (cm)')
plt.ylabel('Długość płatka (cm)')
plt.plot(wPDpDk['Długość działki kielicha (cm)'], rLDpDk_line, color = "r")
plt.show()

# Wykres punktowy (długość płatka i szerokość działki kielicha)
plt.scatter(wPDpSZk['Szerokość działki kielicha (cm)'],wPDpSZk['Długość płatka (cm)'], marker='o')
plt.title("r = " + str(wPDpSZk_value.round(2)) + "; y ="+ str(rLDpSZk_a.round(1))+ "x + " + str(rLDpSZk_b.round(1)))
plt.xlabel('Szerokość działki kielicha (cm)')
plt.ylabel('Długość płatka (cm)')
plt.plot( wPDpSZk['Szerokość działki kielicha (cm)'], rLDpSZk_line, color = "r")
plt.show()

# Wykres punktowy (długość płatka i szerokość płatka)
plt.scatter(wPDpSZp['Długość płatka (cm)'], wPDpSZp['Szerokość płatka (cm)'], marker='o')
plt.title("r = " + str(wPDpSZp_value.round(2)) + "; y ="+ str(rLDpSZp_a.round(1))+ "x - " + str(rLDpSZp_b.round(1)))
plt.xlabel('Długość płatka (cm)')
plt.ylabel('Szerokość płatka (cm)')
plt.plot( wPDpSZp['Długość płatka (cm)'], rLDpSZp_line, color = "r")
plt.show()