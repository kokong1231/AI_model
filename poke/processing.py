import pandas as pd
from pandas.core import indexing

data = pd.read_csv('./pokemon.csv')
data_name = data.sort_values(by=['Name'], axis=0)

data_name.to_csv('./pokemon_sort.csv', index=False)