import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# Yahoo Finance'den veri çekmek için yfinance kütüphanesini kullandık.
import yfinance as yf #verileri çekmek için kütüphaneyi import ediyoruz

# verileri çekme
symbol = "AAPL"  # işlem yapmasını istediğimiz hisse senedinin kodu
start_date = "2022-01-01"
end_date = "2023-01-01"
data = yf.download(symbol, start=start_date, end=end_date)

# kapanış fiyatlarını kullanma
ts = data['Close'].values.reshape(-1, 1)

# Min-Max ölçeklendirme
scaler = MinMaxScaler(feature_range=(0, 1))
ts_scaled = scaler.fit_transform(ts)

# toplanan veriyi LSTM için uygun hale getirme
def create_dataset(X, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:i+time_steps]
        Xs.append(v)
        ys.append(X[i+time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 20  # zaman serisi pencere boyutu
X, y = create_dataset(ts_scaled, time_steps)

# veriyi eğitim ve test setlerine bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LSTM modeli oluşturma (TensorFlow'un düşük seviyeli API kullanılarak)
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=50, activation='relu', input_shape=(time_steps, 1)),
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer='adam', loss='mse')

# oluşturduğumuz modeli eğitme
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1)

# test verisi üzerinde tahmin yapma
y_pred = model.predict(X_test)

# tahminleri ters ölçeklendirme
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# anomalileri tespit ettiğimiz blok
threshold = 0.02  # Değiştirebileceğiniz bir eşik değeri
anomalies = np.where(np.abs(y_test_inv - y_pred_inv.reshape(-1)) > threshold)[0]

# grafikleri ve çizimleri oluşturduğumuz blok
plt.figure(figsize=(14, 8))
plt.plot(data.index[-len(y_test_inv):], y_test_inv, label='Gerçek Veri', color='blue')
plt.plot(data.index[-len(y_test_inv):], y_pred_inv, label='Tahmin', color='green')
plt.scatter(data.index[-len(y_test_inv):][anomalies], y_test_inv[anomalies], label='Anomali', color='red')
plt.title('LSTM ile Anomali Tespiti')
plt.xlabel('Tarih')
plt.ylabel('Kapanış Fiyatı')
plt.legend()
plt.show()
