import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#from sklearn.cluster import KMeans
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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

#print(daneTestowe)
#print(znormalizowaneDaneTestowe)

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

#print(daneTreningowe)
#print(znormalizowaneDaneTreningowe)
#print(xMinDlKielichaTrening, xMaxDlKielichaTrening)

def normaliseTestData(what, xMin, xMax):
    for i in range(len(what)):
        what[i]=(what[i]-xMin)/(xMax-xMin)
    return what

znormalizowaneDaneTestowe['Sepal length']=normaliseTestData(dlKielichaTest, xMinDlKielichaTrening, xMaxDlKielichaTrening)
znormalizowaneDaneTestowe['Sepal width']=normaliseTestData(szerKielichaTest, xMinSzerKielichaTrening, xMaxSzerKielichaTrening)
znormalizowaneDaneTestowe['Petal length']=normaliseTestData(dlPlatkaTest, xMinDlPlatkaTrening, xMaxDlPlatkaTrening)
znormalizowaneDaneTestowe['Petal width']=normaliseTestData(szerPlatkaTest, xMinSzerPlatkaTrening, xMaxSzerPlatkaTrening)

#(daneTestowe)
#print (znormalizowaneDaneTestowe)

#zademonstrować działanie algorytmu k-NN w oparciu o odpowiednio zmodyfikowany zbiór irysów,
# zawierający obiekty z trzech klas: setosa, versicolor i virginica
# uwzględnić normalizację danych
knn=KNeighborsClassifier(n_neighbors=5)
gatunekTreningowy=znormalizowaneDaneTreningowe['Species']
gatunekTestowy=znormalizowaneDaneTestowe['Species']

daneTreningowe = znormalizowaneDaneTreningowe.drop(columns=['Species'])
daneTestowe = znormalizowaneDaneTestowe.drop(columns=['Species'])

knn.fit(daneTreningowe,gatunekTreningowy)
print(gatunekTreningowy)
print(gatunekTestowy)

przewidywanyGatunekTestowy=knn.predict(daneTestowe)
print(gatunekTreningowy)
print(przewidywanyGatunekTestowy)

macierzPomylekKlasyfikacji=confusion_matrix(gatunekTestowy,przewidywanyGatunekTestowy)
print(macierzPomylekKlasyfikacji)
raport=classification_report(gatunekTestowy,przewidywanyGatunekTestowy)
print(raport)

#print(knn.score(znormalizowaneDaneTestowe,gatunekTestowy))