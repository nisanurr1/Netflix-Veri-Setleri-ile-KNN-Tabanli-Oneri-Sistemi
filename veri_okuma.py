 # .csv dosyalarımı okuyup pandas dataframe'lerine çevireceğim kısım 

import pandas as pd

film_df = pd.read_csv("data/netflix_filmler.csv")
dizi_df = pd.read_csv("data/netflix_diziler.csv")

print("Film Veri Seti")
print(film_df.head())

print("\nVeri seti bilgileri:")
print(film_df.info())
# info ile beraber satır sütun sayısı, veri tipleri, eksik veri var mı kontrolü yapıyorum.

print("\nSütun isimleri:")
print(film_df.columns)

print("\nEksik veri sayıları:")
print(film_df.isnull().sum())

# Buraya kadar filmler içindi.

print("Dizi Veri Seti")
print(dizi_df.head())

print("\nVeri seti bilgileri:")
print(dizi_df.info())

print("\nSütun isimleri:")
print(dizi_df.columns)

print("\nEksik veri sayıları:")
print(dizi_df.isnull().sum())

