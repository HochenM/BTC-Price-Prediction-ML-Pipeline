# BTC Price Prediction — ML Pipeline
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# ===== LOAD DATA =====
df = pd.read_csv("BTC-USD.csv")

# ===== CHECK NaN =====
df.isna().sum()

# ===== FEATURE ENGINEERING =====
df["logClose_return"] = np.log(df['Close'] / df['Close'].shift(1))
df.dropna(inplace=True)

# ===== TARGET =====
df["PriceTomorrow"] = df["Close"].shift(-1)
df.dropna(inplace=True)

df["LogReturnTomorrow"] = df["logClose_return"].shift(-1)
df.dropna(inplace=True)

X = df['logClose_return']
y = df["LogReturnTomorrow"]

# ===== TRAIN-TEST SPLIT =====
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

# ===== PLOT SPLIT =====
plt.figure(figsize=(15, 8))
plt.plot(y_train, label="train")
plt.plot(y_test, label="test")
plt.legend()
plt.show()

# ===== SEQUENCING X_train =====
n_samples = X_train.shape[0]
n_sequence = 7
X_train_batch = []
y_train_batch = []

for i in range(0, n_samples - n_sequence):
    batch_train = X_train.iloc[i:i + n_sequence]
    X_train_batch.append(batch_train)
    y_train_new = y_train.iloc[i + n_sequence]
    y_train_batch.append(y_train_new)

print(len(X_train_batch))
print(len(y_train_batch))

# ===== SEQUENCING X_test =====
n_samples = X_test.shape[0]
n_sequence = 7
X_test_batch = []
y_test_batch = []

for i in range(0, n_samples - n_sequence):
    batch_test = X_test.iloc[i:i + n_sequence]
    X_test_batch.append(batch_test)
    y_test_new = y_test.iloc[i + n_sequence]
    y_test_batch.append(y_test_new)

print(len(X_test_batch))
print(len(y_test_batch))

# ===== CONVERT TO ARRAYS =====
X_train_array = np.array(X_train_batch)
X_test_array = np.array(X_test_batch)
y_train_array = np.array(y_train_batch)
y_test_array = np.array(y_test_batch)

print("X_train shape:", X_train_array.shape)
print("X_test shape:", X_test_array.shape)
print("y_train shape:", y_train_array.shape)
print("y_test shape:", y_test_array.shape)

# ===== PLOT ARRAYS =====
plt.figure(figsize=(15, 8))
plt.plot(y_train_array, label="train")
plt.plot(y_test_array, label="test")
plt.legend()
plt.show()

# ===== PIPELINE =====
pipeline = Pipeline([
    ('model', SGDRegressor(max_iter=1000, random_state=42, penalty='l2'))
])

# ===== TRAIN =====
pipeline.fit(X_train_array, y_train_array)
# ===== PREDICT TEST =====
y_pred_test = pipeline.predict(X_test_array)
plt.figure(figsize=(15, 8))
plt.plot(y_test_array, label='real')
plt.plot(y_pred_test, label="predict")
plt.legend()
plt.show()

# ===== CONVERT TO DOLLAR PRICES (TEST) =====
baseline_price = df["Close"].iloc[len(X_train) + n_sequence : len(X_train) + n_sequence + len(y_pred_test)]
predicted_price = baseline_price * np.exp(y_pred_test)

real_price = df["Close"].iloc[len(X_train) + n_sequence + 1 : len(X_train) + n_sequence + 1 + len(y_pred_test)]

predicted_price = predicted_price[:-1]
real_price = real_price.reset_index(drop=True)
predicted_price = predicted_price.reset_index(drop=True)

# ===== PLOT PRICES =====
plt.figure(figsize=(15, 8))
plt.plot(predicted_price, label='Predicted')
plt.plot(real_price, label='Real')
plt.legend()
plt.title('BTC Price Prediction vs Reality')
plt.show()

# ===== EVALUATE TEST =====
mae = mean_absolute_error(real_price, predicted_price)
rmse = np.sqrt(mean_squared_error(real_price, predicted_price))

print(f"Test MAE:  ${mae:,.2f}")
print(f"Test RMSE: ${rmse:,.2f}")

# ===== PREDICT TRAIN =====
y_pred_train = pipeline.predict(X_train_array)

plt.figure(figsize=(15, 8))
plt.plot(y_train_array, label='real')
plt.plot(y_pred_train, label="predict")
plt.legend()
plt.show()

# ===== CONVERT TO DOLLAR PRICES (TRAIN) =====
start_idx_train = n_sequence
baseline_train = df["Close"].iloc[start_idx_train : start_idx_train + len(y_pred_train)]
real_price_train = df["Close"].iloc[start_idx_train + 1 : start_idx_train + 1 + len(y_pred_train)]
predicted_price_train = baseline_train.values * np.exp(y_pred_train)

# ===== EVALUATE TRAIN =====
mae_train_dollar = mean_absolute_error(real_price_train, predicted_price_train)
rmse_train_dollar = np.sqrt(mean_squared_error(real_price_train, predicted_price_train))

print(f"Train MAE:  ${mae_train_dollar:,.2f}")
print(f"Train RMSE: ${rmse_train_dollar:,.2f}")

# ===== PLOT TRAIN PRICES =====
plt.figure(figsize=(15, 8))
plt.plot(predicted_price_train, label='Predicted')
plt.plot(real_price_train.values, label='Real')
plt.legend()
plt.title('BTC Price Prediction — Training Data')
plt.show()