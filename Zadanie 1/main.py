import pandas as pd


dane = pd.read_csv('dane.csv',
                   header = None,
                   names = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Species'])


#Liczności poszczególnych gatunków

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
print(liczności)