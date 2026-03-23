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

FORWARD_DAYS = 5      # predecir si precio sube en 5 días
CONF_THRESHOLD = 0.60 # solo señal si confianza > 60%

def engineer_features(df, forward_days=FORWARD_DAYS):
    d = df.copy()
    # Returns
    d['return_1']  = d['Close'].pct_change()
    d['return_3']  = d['Close'].pct_change(3)
    d['return_5']  = d['Close'].pct_change(5)
    d['return_10'] = d['Close'].pct_change(10)
    d['return_20'] = d['Close'].pct_change(20)
    # Moving averages
    d['ma_5']      = d['Close'].rolling(5).mean()
    d['ma_10']     = d['Close'].rolling(10).mean()
    d['ma_20']     = d['Close'].rolling(20).mean()
    d['ma_50']     = d['Close'].rolling(50).mean()
    d['ma_5_20']   = d['ma_5'] / d['ma_20']
    d['ma_10_50']  = d['ma_10'] / d['ma_50']
    # Volatility
    d['vol_5']     = d['return_1'].rolling(5).std()
    d['vol_20']    = d['return_1'].rolling(20).std()
    d['vol_ratio'] = d['vol_5'] / (d['vol_20'] + 1e-9)
    # Momentum
    d['mom_5']     = d['Close'] / d['Close'].shift(5) - 1
    d['mom_10']    = d['Close'] / d['Close'].shift(10) - 1
    d['mom_20']    = d['Close'] / d['Close'].shift(20) - 1
    # RSI
    delta = d['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = -delta.clip(upper=0).rolling(14).mean()
    d['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-9)))
    # MACD
    ema12 = d['Close'].ewm(span=12).mean()
    ema26 = d['Close'].ewm(span=26).mean()
    d['macd']        = ema12 - ema26
    d['macd_signal'] = d['macd'].ewm(span=9).mean()
    d['macd_diff']   = d['macd'] - d['macd_signal']
    # Bollinger
    bb_mid   = d['Close'].rolling(20).mean()
    bb_std   = d['Close'].rolling(20).std()
    d['bb_pos'] = (d['Close'] - bb_mid) / (bb_std + 1e-9)
    # Volume features (if available)
    if 'Volume' in d.columns:
        d['vol_change'] = d['Volume'].pct_change()
        d['vol_ma5']    = d['Volume'].rolling(5).mean()
        d['vol_ratio2'] = d['Volume'] / (d['vol_ma5'] + 1e-9)
    # Label: sube más del 0% en forward_days días
    d['label'] = (d['Close'].shift(-forward_days) > d['Close']).astype(int)
    d = d.dropna()
    return d

def get_feature_cols(df):
    base = ['return_1','return_3','return_5','return_10','return_20',
            'ma_5_20','ma_10_50','vol_5','vol_20','vol_ratio',
            'mom_5','mom_10','mom_20','rsi','macd','macd_signal',
            'macd_diff','bb_pos']
    if 'vol_change' in df.columns:
        base += ['vol_change','vol_ratio2']
    return base

def build_lstm(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
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
    seq_len = 20

    train_feat = engineer_features(train_df)
    test_feat  = engineer_features(test_df)

    feature_cols = get_feature_cols(train_feat)

    if len(train_feat) < seq_len + 10:
        raise ValueError("Not enough training data for classification.")

    X_train = train_feat[feature_cols].values
    y_train = train_feat['label'].values
    X_test  = test_feat[feature_cols].values
    y_test  = test_feat['label'].values

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # --- LSTM ---
    X_tr_seq, y_tr_seq = make_sequences(X_train_sc, y_train, seq_len)
    X_te_seq, y_te_seq = make_sequences(X_test_sc,  y_test,  seq_len)

    lstm_proba = np.full(len(y_test), 0.5)
    if len(X_tr_seq) > 0 and len(X_te_seq) > 0:
        try:
            tf.keras.backend.clear_session()
            lstm = build_lstm((seq_len, len(feature_cols)))
            es = EarlyStopping(patience=7, restore_best_weights=True)
            lstm.fit(
                X_tr_seq, y_tr_seq,
                epochs=lstm_epochs,
                batch_size=batch_size,
                validation_split=0.1,
                callbacks=[es],
                verbose=0
            )
            lstm_proba_seq = lstm.predict(X_te_seq, verbose=0).flatten()
            offset = len(y_test) - len(lstm_proba_seq)
            lstm_proba[offset:] = lstm_proba_seq
        except Exception as e:
            print(f"LSTM failed: {e}")

    # --- LightGBM ---
    lgbm = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1
    )
    lgbm.fit(X_train_sc, y_train)
    lgbm_proba = lgbm.predict_proba(X_test_sc)[:, 1]

    # --- GBM ---
    gbm = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8
    )
    gbm.fit(X_train_sc, y_train)
    gbm_proba = gbm.predict_proba(X_test_sc)[:, 1]

    # --- Ensemble (weighted: LGBM y GBM pesan más que LSTM en tabular) ---
    ensemble_proba = (lstm_proba * 0.25 + lgbm_proba * 0.40 + gbm_proba * 0.35)

    # Predicción con threshold de confianza
    ensemble_pred = np.where(
        ensemble_proba >= CONF_THRESHOLD, 1,
        np.where(ensemble_proba <= (1 - CONF_THRESHOLD), 0, -1)  # -1 = sin señal
    )
    # Para métricas usar threshold estándar 0.5
    pred_binary = (ensemble_proba >= 0.5).astype(int)

    try:
        auc = roc_auc_score(y_test, ensemble_proba)
    except Exception:
        auc = 0.5

    mae = mean_absolute_error(y_test, ensemble_proba)

    # --- Forecast de precios ---
    last_price = test_df['Close'].iloc[-1]
    avg_move   = test_df['Close'].pct_change().abs().mean()

    # Usar señal del ensemble para dirección
    last_signal = ensemble_proba[-1]
    future_prices = [last_price]
    for i in range(forecast_days):
        if last_signal >= CONF_THRESHOLD:
            direction = 1
        elif last_signal <= (1 - CONF_THRESHOLD):
            direction = -1
        else:
            direction = 0  # sin señal clara → precio plano
        next_price = future_prices[-1] * (1 + direction * avg_move)
        future_prices.append(next_price)
    future_prices = future_prices[1:]

    future_dates = pd.date_range(
        start=test_df['Date'].iloc[-1] + pd.Timedelta(days=1),
        periods=forecast_days,
        freq='B'
    )

    # --- Figura ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: precios + forecast
    historical = pd.concat([train_df, test_df])
    axes[0].plot(historical['Date'], historical['Close'],
                 label='Historical', color='blue', linewidth=1.5)
    axes[0].plot(future_dates, future_prices,
                 label=f'{forecast_days}-Day Forecast', color='red',
                 linestyle='--', linewidth=2)
    axes[0].set_title('Stock Price Forecast (Classification Mode)')
    axes[0].set_ylabel('Price ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: probabilidad ensemble en test
    test_dates = test_feat['Date'] if 'Date' in test_feat.columns else \
                 test_df['Date'].iloc[-len(ensemble_proba):]
    axes[1].plot(test_df['Date'].iloc[-len(ensemble_proba):],
                 ensemble_proba, label='P(UP in 5d)', color='purple', linewidth=1.5)
    axes[1].axhline(CONF_THRESHOLD, color='green', linestyle='--',
                    label=f'Buy threshold ({CONF_THRESHOLD:.0%})')
    axes[1].axhline(1 - CONF_THRESHOLD, color='red', linestyle='--',
                    label=f'Sell threshold ({1-CONF_THRESHOLD:.0%})')
    axes[1].axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    axes[1].set_ylim(0, 1)
    axes[1].set_title('Ensemble Confidence (5-Day Forward)')
    axes[1].set_ylabel('Probability')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    return {
        'auc':          auc,
        'mae':          mae,
        'pred':         pred_binary,
        'y_true':       y_test,
        'proba':        ensemble_proba,
        'forecast':     np.array(future_prices),
        'future_dates': future_dates,
        'fig':          fig
    }
