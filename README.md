#  Car Price Prediction using Linear Regression

Ovaj mini projekt koristi mašinsko učenje (ML) za **predikciju cijene automobila** na osnovu tehničkih karakteristika vozila.

##  Kratak opis

Korištenjem **Linearne regresije**, model uči odnose između faktora kao što su veličina motora, potrošnja goriva, težina i broj cilindara — i predviđa **prosječnu cijenu automobila**.

Projekt je napravljen kao edukativni primjer iz domene **superviziranog učenja**.

---

##  Korištene biblioteke

- `pandas` — učitavanje i obrada podataka
- `matplotlib`, `seaborn` — vizualizacija
- `scikit-learn` — modeliranje, evaluacija i podjela podataka

---

##  Dataset

- **Naziv**: [Cars93_miss.csv](https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv)
- **Sadrži**: 93 automobila, 27 atributa (neki s nedostajućim vrijednostima)
- **Izvor**: `selva86` GitHub dataset repo

---

##  Koraci:

1. Učitavanje i čišćenje podataka (`dropna`)
2. Odabir relevantnih atributa (`features`)
3. Podjela na **train / validation / test** skupove
4. Treniranje modela: `LinearRegression`
5. Evaluacija performansi (`MSE`, `R²`)
6. Vizualizacija predikcija

---

##  Rezultati

- **Validacija R² score**: ~0.5 (može varirati)
- Vizualizacija: scatter plot stvarnih vs predviđenih cijena


---

##  Mogućnosti proširenja

- Zamjena modela: `RandomForestRegressor`, `Ridge`, `SVR`, itd.
- Standardizacija ulaznih vrijednosti (`StandardScaler`)
- `GridSearchCV` za optimizaciju hiperparametara
- Spremanje modela pomoću `joblib` ili `pickle`
- Deploy modela kao REST API (npr. pomoću Flask-a)

---

##  Pokretanje

```bash
pip install -r requirements.txt
python car_price_model.py
