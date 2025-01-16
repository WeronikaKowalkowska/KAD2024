import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

# odczyt danych z plików csv
daneTestowe = pd.read_csv('data3_test.csv',
                          header=None,
                          names=['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Species'])

daneTreningowe = pd.read_csv('data3_train.csv',
                             header=None,
                             names=['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Species'])

# normalizacja danych
znormalizowaneDaneTestowe = daneTestowe.copy()
znormalizowaneDaneTreningowe = daneTreningowe.copy()

# print(daneTestowe)
# print(znormalizowaneDaneTestowe)

dlKielichaTrening = np.array(znormalizowaneDaneTreningowe['Sepal length'])
szerKielichaTrening = np.array(znormalizowaneDaneTreningowe['Sepal width'])
dlPlatkaTrening = np.array(znormalizowaneDaneTreningowe['Petal length'])
szerPlatkaTrening = np.array(znormalizowaneDaneTreningowe['Petal width'])

dlKielichaTest = np.array(znormalizowaneDaneTestowe['Sepal length'])
szerKielichaTest = np.array(znormalizowaneDaneTestowe['Sepal width'])
dlPlatkaTest = np.array(znormalizowaneDaneTestowe['Petal length'])
szerPlatkaTest = np.array(znormalizowaneDaneTestowe['Petal width'])


def normaliseTrainingData(what):
    xMin = min(what)
    xMax = max(what)
    for i in range(len(what)):
        what[i] = (what[i] - xMin) / (xMax - xMin)
    return what, xMin, xMax


znormalizowaneDaneTreningowe['Sepal length'], xMinDlKielichaTrening, xMaxDlKielichaTrening = normaliseTrainingData(
    dlKielichaTrening)
znormalizowaneDaneTreningowe['Sepal width'], xMinSzerKielichaTrening, xMaxSzerKielichaTrening = normaliseTrainingData(
    szerKielichaTrening)
znormalizowaneDaneTreningowe['Petal length'], xMinDlPlatkaTrening, xMaxDlPlatkaTrening = normaliseTrainingData(
    dlPlatkaTrening)
znormalizowaneDaneTreningowe['Petal width'], xMinSzerPlatkaTrening, xMaxSzerPlatkaTrening = normaliseTrainingData(
    szerPlatkaTrening)

# print(daneTreningowe)
# print(znormalizowaneDaneTreningowe)
#print(xMinDlKielichaTrening, xMaxDlKielichaTrening)

def normaliseTestData(what, xMin, xMax):
    for i in range(len(what)):
        what[i] = (what[i] - xMin) / (xMax - xMin)
    return what


znormalizowaneDaneTestowe['Sepal length'] = normaliseTestData(dlKielichaTest, xMinDlKielichaTrening,
                                                              xMaxDlKielichaTrening)
znormalizowaneDaneTestowe['Sepal width'] = normaliseTestData(szerKielichaTest, xMinSzerKielichaTrening,
                                                             xMaxSzerKielichaTrening)
znormalizowaneDaneTestowe['Petal length'] = normaliseTestData(dlPlatkaTest, xMinDlPlatkaTrening, xMaxDlPlatkaTrening)
znormalizowaneDaneTestowe['Petal width'] = normaliseTestData(szerPlatkaTest, xMinSzerPlatkaTrening,
                                                             xMaxSzerPlatkaTrening)
# print(daneTestowe)
# print(znormalizowaneDaneTestowe)


