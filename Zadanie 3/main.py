import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#from sklearn.cluster import KMeans
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

#odczyt danych z plików csv
daneTestowe = pd.read_csv('data3_test.csv',
                   header = None,
                   names = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Species'])

daneTreningowe = pd.read_csv('data3_train.csv',
                          header = None,
                          names = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Species'])

#normalizacja danych
znormalizowaneDaneTestowe=daneTestowe.copy()
znormalizowaneDaneTreningowe=daneTreningowe.copy()

dlKielichaTrening = np.array(znormalizowaneDaneTreningowe['Sepal length'])
szerKielichaTrening = np.array(znormalizowaneDaneTreningowe['Sepal width'])
dlPlatkaTrening = np.array(znormalizowaneDaneTreningowe['Petal length'])
szerPlatkaTrening = np.array(znormalizowaneDaneTreningowe['Petal width'])

dlKielichaTest = np.array(znormalizowaneDaneTestowe['Sepal length'])
szerKielichaTest = np.array(znormalizowaneDaneTestowe['Sepal width'])
dlPlatkaTest = np.array(znormalizowaneDaneTestowe['Petal length'])
szerPlatkaTest = np.array(znormalizowaneDaneTestowe['Petal width'])

def normaliseTrainingData(what):
    xMin=min(what)
    xMax=max(what)
    for i in range(len(what)):
        what[i]=(what[i]-xMin)/(xMax-xMin)
    return what, xMin, xMax

znormalizowaneDaneTreningowe['Sepal length'], xMinDlKielichaTrening, xMaxDlKielichaTrening=normaliseTrainingData(dlKielichaTrening)
znormalizowaneDaneTreningowe['Sepal width'], xMinSzerKielichaTrening, xMaxSzerKielichaTrening=normaliseTrainingData(szerKielichaTrening)
znormalizowaneDaneTreningowe['Petal length'], xMinDlPlatkaTrening, xMaxDlPlatkaTrening=normaliseTrainingData(dlPlatkaTrening)
znormalizowaneDaneTreningowe['Petal width'], xMinSzerPlatkaTrening, xMaxSzerPlatkaTrening=normaliseTrainingData(szerPlatkaTrening)

def normaliseTestData(what, xMin, xMax):
    for i in range(len(what)):
        what[i]=(what[i]-xMin)/(xMax-xMin)
    return what

znormalizowaneDaneTestowe['Sepal length']=normaliseTestData(dlKielichaTest, xMinDlKielichaTrening, xMaxDlKielichaTrening)
znormalizowaneDaneTestowe['Sepal width']=normaliseTestData(szerKielichaTest, xMinSzerKielichaTrening, xMaxSzerKielichaTrening)
znormalizowaneDaneTestowe['Petal length']=normaliseTestData(dlPlatkaTest, xMinDlPlatkaTrening, xMaxDlPlatkaTrening)
znormalizowaneDaneTestowe['Petal width']=normaliseTestData(szerPlatkaTest, xMinSzerPlatkaTrening, xMaxSzerPlatkaTrening)

#zademonstrować działanie algorytmu k-NN dla k od 1 do 15 w oparciu o odpowiednio zmodyfikowany zbiór irysów,
#zawierający obiekty z trzech klas: setosa, versicolor i virginica;
#uwzględnić normalizację danych

dokladnoscWynikiWprocentach = []
kRange = np.arange(1, 16, 1)

najlepszeK = None
najlepszaDokladnosc = 0
macierzPomylekKlasyfikacjiNajlepszeK = None

gatunekTreningowy=znormalizowaneDaneTreningowe['Species']
gatunekTestowy=znormalizowaneDaneTestowe['Species']

znormalizowaneDaneTreningowe=znormalizowaneDaneTreningowe.drop(columns=['Species'])
znormalizowaneDaneTestowe=znormalizowaneDaneTestowe.drop(columns=['Species'])

for k in kRange:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(znormalizowaneDaneTreningowe, gatunekTreningowy)
    przewidywanyGatunekTestowy = knn.predict(znormalizowaneDaneTestowe)
    dokladnosc = accuracy_score(gatunekTestowy, przewidywanyGatunekTestowy)
    dokladnoscWynikiWprocentach.append(dokladnosc * 100)

    if dokladnosc > najlepszaDokladnosc:
        najlepszeK = k
        najlepszaDokladnosc = dokladnosc
        macierzPomylekKlasyfikacjiNajlepszeK = confusion_matrix(gatunekTestowy,przewidywanyGatunekTestowy)


plt.bar(kRange, dokladnoscWynikiWprocentach)
plt.xticks(kRange)
# plt.yticks(np.arange(90, 102, 2))
plt.xlabel("Liczba sąsiadów (k)")
plt.ylabel("Dokładność (%)")
plt.title("Dokładność klasyfikacji k-NN w zależności od liczby sąsiadów")
# plt.ylim(90, 100)
plt.show()

#macierz pomyłek dla klasyfikatora o największej dokładności - na podstawie 4 cech
disp = ConfusionMatrixDisplay(confusion_matrix=macierzPomylekKlasyfikacjiNajlepszeK, display_labels=['Setosa', 'Versicolor', 'Virginica'])
disp.plot(cmap='Blues', colorbar=False)
plt.title("Mecierz pomyłek dla klasyfikatora o największej dokładności (k = " + str(najlepszeK) + " )\n na podstawie czterech cech")
plt.xlabel("Rozpoznana klasa")
plt.ylabel('Faktyczna klasa')
plt.show()

#algorytm k-NN dla par cech

def knnForPairs(tableTren, tableTest, gatunekTreningowy, gatunekTestowy, kRange):
    najlepszeK = None
    najlepszaDokladnosc = 0
    dokladnoscWynikiWprocentach = []
    macierzPomylekKlasyfikacjiNajlepszeK = None

    for k in kRange:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(tableTren, gatunekTreningowy)
        przewidywanyGatunekTestowy = knn.predict(tableTest)
        dokladnosc = accuracy_score(gatunekTestowy, przewidywanyGatunekTestowy)
        dokladnoscWynikiWprocentach.append(dokladnosc * 100)

        if dokladnosc > najlepszaDokladnosc:
            najlepszeK = k
            najlepszaDokladnosc = dokladnosc
            macierzPomylekKlasyfikacjiNajlepszeK = confusion_matrix(gatunekTestowy, przewidywanyGatunekTestowy)

    return dokladnoscWynikiWprocentach, macierzPomylekKlasyfikacjiNajlepszeK, najlepszeK

def plotAndMatrix(kRange, dokladnoscWynikiWprocentach, macierzPomylek, najlepszeK, string):
    plt.bar(kRange, dokladnoscWynikiWprocentach)
    plt.xticks(kRange)
    plt.xlabel("Liczba sąsiadów (k)")
    plt.ylabel("Dokładność (%)")
    plt.title("Dokładność klasyfikacji k-NN w zależności od liczby sąsiadów\n" + string)
    plt.ylim(65, 100)
    plt.show()

    disp = ConfusionMatrixDisplay(confusion_matrix=macierzPomylek, display_labels=['Setosa', 'Versicolor', 'Virginica'])
    disp.plot(cmap='Blues', colorbar=False)
    plt.title("Mecierz pomyłek dla klasyfikatora o największej dokładności (k = " + str(najlepszeK) + " )\n" + string)
    plt.xlabel("Rozpoznana klasa")
    plt.ylabel('Faktyczna klasa')
    plt.show()


#Długość działki kielicha, Szerokość działki kielicha
dlKielichaSzerKielichaTereningowy = znormalizowaneDaneTreningowe[['Sepal length', 'Sepal width']]
dlKielichaSzerKielichaTestowy = znormalizowaneDaneTestowe[['Sepal length', 'Sepal width']]
dlKielichaSzerKielichaString = "(Długość działki kielicha, Szerokość działki kielicha)"

dokladnoscDlKielichaSzerKielicha, macierzPomylekDlKielichaSzerKielicha, najlepszeKDlKielichaSzerKielicha = knnForPairs(dlKielichaSzerKielichaTereningowy, dlKielichaSzerKielichaTestowy, gatunekTreningowy, gatunekTestowy, kRange)
plotAndMatrix(kRange, dokladnoscDlKielichaSzerKielicha, macierzPomylekDlKielichaSzerKielicha, najlepszeKDlKielichaSzerKielicha, dlKielichaSzerKielichaString)

#Długość działki kielicha, Szerokość płatka
dlKielichaSzerPlatkaTereningowy = znormalizowaneDaneTreningowe[['Sepal length', 'Petal width']]
dlKielichaSzerPlatkaTestowy = znormalizowaneDaneTestowe[['Sepal length', 'Petal width']]
dlKielichaSzerPlatkaString = "(Długość działki kielicha, Szerokość płatka)"

dokladnoscDlKielichaSzerPlatka, macierzPomylekDlKielichaSzerPlatka, najlepszeKDlKielichaSzerPlatka = knnForPairs(dlKielichaSzerPlatkaTereningowy, dlKielichaSzerPlatkaTestowy, gatunekTreningowy, gatunekTestowy, kRange)
plotAndMatrix(kRange, dokladnoscDlKielichaSzerPlatka, macierzPomylekDlKielichaSzerPlatka, najlepszeKDlKielichaSzerPlatka, dlKielichaSzerPlatkaString)
