import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

def engineer_features(df):
    d = df.copy()
    d['return_1'] = d['Close'].pct_change()
    d['return_5'] = d['Close'].pct_change(5)
    d['return_10'] = d['Close'].pct_change(10)
    d['ma_5'] = d['Close'].rolling(5).mean()
    d['ma_20'] = d['Close'].rolling(20).mean()
    d['ma_ratio'] = d['ma_5'] / d['ma_20']
    d['volatility'] = d['return_1'].rolling(10).std()
    d['momentum'] = d['Close'] / d['Close'].shift(10) - 1
    delta = d['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    d['rsi'] = 100 - (100 / (1 + rs))
    d['label'] = (d['Close'].shift(-1) > d['Close']).astype(int)
    d = d.dropna()
    return d

def build_lstm(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def make_sequences(X, y, seq_len=20):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

def train_classification_sota(train_df, test_df, lstm_epochs=50, batch_size=16, forecast_days=15):
    feature_cols = ['return_1','return_5','return_10','ma_ratio','volatility','momentum','rsi']
    seq_len = 20

    train_feat = engineer_features(train_df)
    test_feat = engineer_features(test_df)

    if len(train_feat) < seq_len + 10:
        raise ValueError("Not enough training data for classification.")

    X_train = train_feat[feature_cols].values
    y_train = train_feat['label'].values
    X_test = test_feat[feature_cols].values
    y_test = test_feat['label'].values

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # --- LSTM ---
    X_tr_seq, y_tr_seq = make_sequences(X_train_sc, y_train, seq_len)
    X_te_seq, y_te_seq = make_sequences(X_test_sc, y_test, seq_len)

    lstm_proba = np.full(len(y_test), 0.5)
    if len(X_tr_seq) > 0 and len(X_te_seq) > 0:
        try:
            tf.keras.backend.clear_session()
            lstm = build_lstm((seq_len, len(feature_cols)))
            es = EarlyStopping(patience=5, restore_best_weights=True)
            lstm.fit(
                X_tr_seq, y_tr_seq,
                epochs=lstm_epochs,
                batch_size=batch_size,
                validation_split=0.1,
                callbacks=[es],
                verbose=0
            )
            lstm_proba_seq = lstm.predict(X_te_seq, verbose=0).flatten()
            # Align to full test length
            offset = len(y_test) - len(lstm_proba_seq)
            lstm_proba[offset:] = lstm_proba_seq
        except Exception as e:
            print(f"LSTM failed: {e}")

    # --- LightGBM ---
    lgbm = LGBMClassifier(n_estimators=200, learning_rate=0.05, verbose=-1)
    lgbm.fit(X_train_sc, y_train)
    lgbm_proba = lgbm.predict_proba(X_test_sc)[:, 1]

    # --- GBM ---
    gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05)
    gbm.fit(X_train_sc, y_train)
    gbm_proba = gbm.predict_proba(X_test_sc)[:, 1]

    # --- Ensemble ---
    ensemble_proba = (lstm_proba + lgbm_proba + gbm_proba) / 3.0
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)

    try:
        auc = roc_auc_score(y_test, ensemble_proba)
    except Exception:
        auc = 0.5

    mae = mean_absolute_error(y_test, ensemble_proba)

    # --- Price forecast using last known price + predicted direction ---
    last_price = test_df['Close'].iloc[-1]
    avg_move = test_df['Close'].pct_change().abs().mean()
    future_prices = [last_price]
    for i in range(forecast_days):
        direction = 1 if ensemble_proba[-1] >= 0.5 else -1
        next_price = future_prices[-1] * (1 + direction * avg_move)
        future_prices.append(next_price)
    future_prices = future_prices[1:]

    future_dates = pd.date_range(
        start=test_df['Date'].iloc[-1] + pd.Timedelta(days=1),
        periods=forecast_days,
        freq='B'
    )

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(12, 5))
    historical = pd.concat([train_df, test_df])
    ax.plot(historical['Date'], historical['Close'], label='Historical', color='blue', linewidth=1.5)
    ax.plot(future_dates, future_prices, label='Forecast', color='red', linestyle='--', linewidth=2)
    ax.set_title('Stock Price Forecast (Classification Mode)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return {
        'auc': auc,
        'mae': mae,
        'pred': ensemble_pred,
        'y_true': y_test,
        'proba': ensemble_proba,
        'forecast': np.array(future_prices),
        'future_dates': future_dates,
        'fig': fig
    }
