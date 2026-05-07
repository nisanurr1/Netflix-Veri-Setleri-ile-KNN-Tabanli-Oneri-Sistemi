from flask import Flask, render_template, request
from knn_model import film_oner, dizi_oner

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def ana_sayfa():
    oneriler = None
    aranan_icerik = ""
    secilen_tur = ""

    if request.method == "POST":
        aranan_icerik = request.form["icerik_adi"]
        secilen_tur = request.form["icerik_turu"]

        if secilen_tur == "film":
            oneriler = film_oner(aranan_icerik)
        elif secilen_tur == "dizi":
            oneriler = dizi_oner(aranan_icerik)

    return render_template(
        "index.html",
        oneriler=oneriler,
        aranan_icerik=aranan_icerik,
        secilen_tur=secilen_tur
    )

if __name__ == "__main__":
    app.run(debug=True)