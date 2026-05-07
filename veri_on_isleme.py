# eksik verileri silme, gereksiz sütunları kaldırma ve gerektiği yerde tür dönüşümlerini yaptığım kısım.

import pandas as pd

film_df = pd.read_csv("data/netflix_filmler.csv")
dizi_df = pd.read_csv("data/netflix_diziler.csv")

# film veri setinde gereksiz ve fazla eksik olan verilerin temizlenmesi
# duration'da(süre) 16000 adet boş satır bulunmakta. 
# netflixe eklenme yılı son eklenenleri görmek açısından sonradan önemli olabilir.
film_df = film_df.drop(columns=["show_id","duration"])
dizi_df = dizi_df.drop(columns=["show_id"])

# İçerik tabanlı öneri için kullanılacak metinsel sütunlar
metinsel_sutunlar = [
    "genres",
    "description",
    "cast",
    "director",
    "country",
    "language"
]

# Eksik metinsel verileri boş metin ile dolduruyoruz
# Burada bunları silmek yerine boşlukla doldurma sebebim eksik veri etkisini azaltmak.
film_df[metinsel_sutunlar] = film_df[metinsel_sutunlar].fillna("")
dizi_df[metinsel_sutunlar] = dizi_df[metinsel_sutunlar].fillna("")

# Veri setlerinin güncel boyutlarını görmek için;
# print("Film Veri Seti Temizlenmiş Yeni Boyutu:")
# print(film_df.shape)
# print("Dizi Veri Seti Temizlenmiş Yeni Boyutu:")
# print(dizi_df.shape)