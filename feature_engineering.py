# knn modeli için önemli metinsel verileri sayısal verilere dönüştüreceğim bölüm.
# knn algoritması sayısal olmayan metinsel veriler ile uzaklık hesaplayamaz bundan dolayı verileri vektörlere çevirmemiz gerekiyor.

import pandas as pd
from veri_on_isleme import film_df, dizi_df
from sklearn.feature_extraction.text import TfidfVectorizer
# bu bizim metinsel ifadelerimizin sayısal vektörlere dönüştürülmesini sağlayacak olann kütüphane.

pd.set_option("display.max_colwidth", None) # içeriklerin devamını görebilmek için geçici olarak ekledim.

# Film veri seti için önemli değişkenlerle içerik sütunu oluşturma
# burada her film için ve aşağıda her dizi için tür, tanım, oyuncu kadrosu, yönetmen, ülke ve dil bilgilerini birleştirerek tek bir içerik sütunu oluşturuyorum.
# içerik tabanlı olduğu için bu şekilde.

film_df["icerik"] = (
    film_df["genres"] + " " + # tür
    film_df["description"] + " " + # tanım
    film_df["cast"] + " " + # oyuncu kadrosu
    film_df["director"] + " " + # yönetmen
    film_df["country"] + " " +
    film_df["language"]
)

# Dizi veri seti için içerik sütunu oluşturma
dizi_df["icerik"] = (
    dizi_df["genres"] + " " +
    dizi_df["description"] + " " +
    dizi_df["cast"] + " " +
    dizi_df["director"] + " " +
    dizi_df["country"] + " " +
    dizi_df["language"]
)

# kontroller içindi
# print("Film içerik örneği:")
# print(film_df[["title", "icerik"]].head())

# print("\nDizi içerik örneği:")
# print(dizi_df[["title", "icerik"]].head())

# Sırada elde ettiğimiz içerikteki metinleri sayısallaştırmak için vektörleştirme işlemi var.
# Bunun için TfidfVectorizer kullanarak içerik sütununu sayısal vektörlere dönüştüreceğim.

film_tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    min_df=2,
    max_df=0.8
)
# burada amacım her içerikte geçen kelimelerin önem ağrılıklarını hesaplamaktı.
# bundan dolayı yaygın kelimlerin etkisini azaltmaya çalıştım.

dizi_tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    min_df=2,
    max_df=0.8
)
# stop_words="english" parametresi, İngilizce'deki yaygın kullanılan kelimeleri (örneğin "the", "is", "and" gibi) vektörleştirme işleminden çıkarır. 
# Bu, modelin daha anlamlı ve ayırt edici özelliklere odaklanmasına yardımcı olur.
# max_features=10000 parametresi, vektörleştirme işleminde kullanılacak maksimum kelime sayısını sınırlar. 
# Bu sayede normalde 65000 kadar kelime içeren içerik sütununu daha yönetilebilir bir boyuta indirgeriz.
# Bunu yapmamış olsaydık maliyetli ve aşırı karmaşık bir modelle karşılaşabilirdik.

# min_df=2 parametresi, vektörleştirme işleminde sadece en az 2 belgede geçen kelimeleri dikkate alır. Bu, nadir kelimelerin etkisini azaltarak modelin genelleme yeteneğini artırır.
# max_df=0.8 parametresi, vektörleştirme işleminde belgelerin %80'inden fazla geçen kelimeleri almaz. Bu da çok yaygın kelimelerin etkisini azaltarak modelin daha anlamlı özelliklere odaklanmasını sağlar. 


# burada içerikler kelime önem ağırlıklarını temsil eden matematiksel vektörler halinde ifade edilmiş oldu 
# ve KNN algoritmasının benzerlik hesaplaması yapabileceği hale geldi.

film_vektorleri = film_tfidf.fit_transform(film_df["icerik"])
dizi_vektorleri = dizi_tfidf.fit_transform(dizi_df["icerik"])
# fit_transform() metodu, önce verilen metin verisinden bir kelime dağarcığı oluşturur (fit) 
# ve ardından her metni bu kelime dağarcığına göre sayısal vektörlere dönüştürür (transform).

# print("Film vektör boyutu:")
# print(film_vektorleri.shape)

# print("\nDizi vektör boyutu:")
# print(dizi_vektorleri.shape)