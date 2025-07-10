import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

url="https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv"

df=pd.read_csv(url)

print(" Prvih 5 redova:")

print(df.head())

print(" Dimenzije:", df.shape)
print(" Nedostajuce vrijednosti po koloni: ")
print(df.isnull().sum())

cols_to_use = [
    "EngineSize", "Horsepower", "RPM", 
    "Cylinders", "MPG.city", "MPG.highway", 
    "Fuel.tank.capacity", "Weight", "Length",
    "Width", "Price"
]

df_clean = df[cols_to_use].dropna()

X=df_clean.drop("Price", axis=1)

y=df_clean["Price"]

print("Velicina X:", X.shape)
print("Velicina y:", y.shape)

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_val_pred = model.predict(X_val)

mse_val = mean_squared_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)

print(f"Validacija - MSE: {mse_val:.2f}")
print(f"Validacija - R²: {r2_val:.2f}")

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_val, y=y_val_pred)
plt.xlabel("Stvarna cijena (Price)")
plt.ylabel("Predvidjena cijena")
plt.title("Validacija: Stvarno vs Predvidjeno")
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.grid(True)
plt.show()

