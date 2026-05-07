# ekik verileri silme, gereksiz sütunları kaldırma ve gerektiği yerde tür dönüşümlerini yaptığım kısım.
# knn algoritması sayısal verilerle çalışır.

import pandas as pd

film_df = pd.read_csv("data/netflix_filmler.csv")
dizi_df = pd.read_csv("data/netflix_diziler.csv")

# film veri setinde gereksiz ve fazla eksik olan verilerin temizlenmesi
# duration'da(süre) 16000 adet boş satır bulunmakta. 
# netflixe eklenme yılı son eklenenleri görmek açısından sonradan önemli olabilir.
film_df = film_df.drop(columns=["show_id","duration"])

# türler ve açıklamalar knn için önemli değişkenler, bu yüzden eksik olan verilerin bulunduğu satırları temizliyorum.
film_df = film_df.dropna(subset=["genres", "description"])

# aynı şekilde dizi veri setinde de tür ve açıklamalar önemli olacak.
dizi_df = dizi_df.dropna(subset=["genres", "description"])
dizi_df = dizi_df.drop(columns=["show_id"])

print("Film Veri Seti Temizlenmiş Yeni Boyutu:")
print(film_df.shape)
print("Dizi Veri Seti Temizlenmiş Yeni Boyutu:")
print(dizi_df.shape)