import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dane = pd.read_csv('data1.csv',
                   header = None,
                   names = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width'])

dlKielicha = np.array(dane['Sepal length'])
szerKielicha = np.array(dane['Sepal width'])
dlPlatka = np.array(dane['Petal length'])
szerPlatka = np.array(dane['Petal width'])
