import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Membaca dataset
file_path = "./estimasi.csv"
data = pd.read_csv(file_path)

# Memisahkan fitur (T1-T6) dan target (Nilai Akhir)
X = data[["T1", "T2", "T3", "T4", "T5", "T6"]]
y = data["Nilai Akhir"]

# K-Fold Cross-Validation dengan 10 fold
kf = KFold(n_splits=10, shuffle=True, random_state=42)

mse_scores = []
loss_scores = []
fold = 1

# MLPRegressor dengan parameter yang disederhanakan
model = MLPRegressor(
    hidden_layer_sizes=(5, ),
    activation='relu',
    solver='adam',
    learning_rate_init=0.01,
    max_iter=5000,
    random_state=42
)

print("\nHasil Validasi 10-Fold:")
for train_index, test_index in kf.split(X):
    # Membagi data menjadi training dan testing
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Melatih model
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)
    
    # Menyimpan nilai loss pada setiap fold
    loss_scores.append(model.loss_)

    print(f"Fold {fold}: MSE = {mse:.4f}, Loss = {model.loss_:.4f}")
    fold += 1

# Menampilkan bobot hasil pelatihan
print("\nBobot Tiap Fitur (T1-T6):")
for i, coef in enumerate(model.coefs_[0][:, 0]):
    print(f"T{i+1}: {coef:.4f}")

# Rata-rata MSE dan Loss
average_mse = np.mean(mse_scores)
average_loss = np.mean(loss_scores)
print(f"\nRata-rata MSE dari 10-Fold Validation: {average_mse:.4f}")
print(f"Rata-rata Loss dari 10-Fold Validation: {average_loss:.4f}")