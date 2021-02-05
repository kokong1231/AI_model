import pandas as pd

data = pd.read_csv('./pokemon_sort.csv')
data_name = data.sort_values(by=['Name'], axis=0)


print(len(data_name['Name']))
# data_name.to_csv('./pokemon_sort.csv', index=False)