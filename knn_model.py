# knn algoritmasıyla ilgili modeli eğitme ve tahmin aşamlarını yürüttüğüm kod kısmı.

from sklearn.neighbors import NearestNeighbors
from feature_engineering import film_vektorleri, dizi_vektorleri, film_df, dizi_df
# burada daha önce oluşturduğumuz TF-IDF vektörlerini knn_model.py dosyasına dahil ediyorum.

film_knn_modeli = NearestNeighbors(
    metric="cosine", # kosinüs benzerliğini, yüksek boyutlu metin verilerinde daha iyi performans gösterdiği için tercih ettim.
    algorithm="brute", # bu film diğer filmlerle ne kadar benziyor hesaplaması yaparken brute-force yöntemini kullandım.
    n_neighbors=51 # hibrit puanlama sistemiyle birlikte 50 öneri göstermek istediğim için 51 komşu seçtim. (en yakın komşu kendisi olacağı için)
)

# metrik olarak TF-IDF ile en uyumlu olan yön benzerliğini tercih ettim.-> Cosine Similarity
# kelime yapıları ne kadar benzerse cosine similarity o kadar yüksek çıkar.
# Burada öklid yüksek boyutlu metin verilerinde daha düşük performans göstereceği için tercih etmedim. 

dizi_knn_modeli = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=51
)
# film/dizi vektörleri uzayında hangi film/dizi nerede bulunuyor modelimiz bunu öğrenecek.
film_knn_modeli.fit(film_vektorleri)
dizi_knn_modeli.fit(dizi_vektorleri)

# burada aday içerikler oluştu.
# projenin isterleri tamamlandı.



def film_oner(film_adi):
    film_adi = film_adi.lower()

    film_eslesme = film_df[film_df["title"].str.lower() == film_adi]

    if film_eslesme.empty:
        return "Film bulunamadı."

    film_indeksi = film_eslesme.index[0]


    mesafeler, indeksler = film_knn_modeli.kneighbors(
        film_vektorleri[film_indeksi],
        n_neighbors=51
    )

    aday_indeksler = indeksler[0][1:]
    aday_mesafeler = mesafeler[0][1:]

    aday_filmler = film_df.iloc[aday_indeksler][
        ["title", "genres", "popularity", "vote_average", "vote_count"]
    ].copy()

    aday_filmler["benzerlik_skoru"] = 1 - aday_mesafeler

    aday_filmler = hibrit_puan_hesapla(aday_filmler)

    aday_filmler = aday_filmler.sort_values(
        by="hibrit_puan",
        ascending=False
    )

    return aday_filmler.head(5)


def dizi_oner(dizi_adi):

    # Büyük-küçük harf problemini önlemek için
    dizi_adi = dizi_adi.lower()

    # Girilen dizi adına karşılık gelen satırı bul
    dizi_eslesme = dizi_df[
        dizi_df["title"].str.lower() == dizi_adi
    ]

    # Eğer dizi bulunamazsa
    if dizi_eslesme.empty:
        return "Dizi bulunamadı."

    # Dizinin index numarasını al
    dizi_indeksi = dizi_eslesme.index[0]

    # En yakın komşuları bul
    mesafeler, indeksler = dizi_knn_modeli.kneighbors(
        dizi_vektorleri[dizi_indeksi],
        n_neighbors=51
    )

    aday_indeksler = indeksler[0][1:]
    aday_mesafeler = mesafeler[0][1:]

    aday_diziler = dizi_df.iloc[aday_indeksler][
        ["title", "genres", "popularity", "vote_average", "vote_count"]
    ].copy()
    # burada "başlık", "türler", "popülerlik", "oy ortalaması", "oy sayısı" bilgileriyle hibrit puan hesabı 
    

    aday_diziler["benzerlik_skoru"] = 1 - aday_mesafeler

    aday_diziler = hibrit_puan_hesapla(aday_diziler)

    aday_diziler = aday_diziler.sort_values(
        by="hibrit_puan",
        ascending=False
    )

    return aday_diziler.head(5)

# Sadece benzer içerikler değil aynı zamanda kalite/ popülerlik bakımından da sıralansın istediğim için hibrit sistemleri araştırdım.
def hibrit_puan_hesapla(veri):
    veri = veri.copy()

    veri["popularity_norm"] = veri["popularity"] / veri["popularity"].max()
    veri["vote_average_norm"] = veri["vote_average"] / veri["vote_average"].max()
    veri["vote_count_norm"] = veri["vote_count"] / veri["vote_count"].max()

    # hangi içeriklerin ne kadar baskın olmasını istiyorsak buradan değişiklikler yapabiliriz.
    veri["hibrit_puan"] = (
        veri["benzerlik_skoru"] * 0.5 +
        veri["popularity_norm"] * 0.3 +
        veri["vote_average_norm"] * 0.4 +
        veri["vote_count_norm"] * 0.3
    )
    # burada son aşamada yaptığım testlere baktığımda KNN'in sadece aday seçtiğini fakat sıralamaya katılmadığını tespit ettim.
    # bundan kaynaklı olarak hibrit puana "benzerlik skoru" eklemesi yaptım.
    # bu sayede artık hem benzer hem de kaliteli içerikler yukarı çıkmış olacak.

    return veri
