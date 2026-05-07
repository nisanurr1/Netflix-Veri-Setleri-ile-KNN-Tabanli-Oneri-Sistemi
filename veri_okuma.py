 # .csv dosyalarımı okuyup pandas dataframe'lerine çevireceğim kısım 

import pandas as pd

film_df = pd.read_csv("data/netflix_filmler.csv")
dizi_df = pd.read_csv("data/netflix_diziler.csv")

print(film_df.head())
print(dizi_df.head())