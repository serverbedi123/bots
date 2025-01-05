import pandas as pd
import numpy as np
import json
import joblib
from binance.client import Client
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import pandas_ta as ta
import asyncio
from telegram import Bot

# Config dosyasını yükle
with open("config.json") as f:
    config = json.load(f)

# Binance API bilgileri
api_key = config["api_key"]
api_secret = config["api_secret"]

# Telegram API bilgileri
telegram_bot_token = config["telegram_bot_token"]
chat_id = config["telegram_chat_id"]

# Binance bağlantısı
client = Client(api_key, api_secret)

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parametreler
symbols = config["symbols"]
leverage = config["leverage"]
risk_per_trade = config["risk_per_trade"]
timeframe = config["timeframe"]
limit = config["limit"]

# Telegram bildirim fonksiyonu
def send_telegram_message(message):
    bot = Bot(token=telegram_bot_token)
    bot.send_message(chat_id=chat_id, text=message)

def calculate_indicators(df):
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['MACD'], df['MACDSignal'], df['MACDHist'] = ta.macd(df['close'], fast=12, slow=26, signal=9)
    bb = ta.bbands(df['close'], length=20, std=2)
    df['UpperBand'] = bb['BBU_20_2.0']
    df['MiddleBand'] = bb['BBM_20_2.0']
    df['LowerBand'] = bb['BBL_20_2.0']
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['Momentum'] = df['close'].diff(10)
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX'] = adx['ADX_14']
    df['DI+'] = adx['DMP_14']
    df['DI-'] = adx['DMN_14']
    df['OBV'] = ta.obv(df['close'], df['volume'])
    stoch_rsi = ta.stochrsi(df['close'], length=14)
    df['StochRSI_K'] = stoch_rsi['STOCHRSIk_14_14_3_3']
    df['StochRSI_D'] = stoch_rsi['STOCHRSId_14_14_3_3']

    # NaN değerleri doldur
    df.fillna(0, inplace=True)

    return df

def detect_candlestick_patterns(df):
    # İkili Tepe Formasyonu
    df['DoubleTop'] = 0
    for i in range(1, len(df) - 1):
        if df['high'].iloc[i - 1] > df['high'].iloc[i] and df['high'].iloc[i] < df['high'].iloc[i + 1]:
            df.at[i, 'DoubleTop'] = 1
            
    # İkili Dip Formasyonu
    df['DoubleBottom'] = 0
    for i in range(1, len(df) - 1):
        if df['low'].iloc[i - 1] < df['low'].iloc[i] and df['low'].iloc[i] > df['low'].iloc[i + 1]:
            df.at[i, 'DoubleBottom'] = 1

    # OBO (Omuz Baş Omuz)
    df['HeadShoulders'] = 0
    for i in range(20, len(df) - 20):
        left_shoulder = df['low'].iloc[i - 20] < df['low'].iloc[i - 10]
        head = df['low'].iloc[i] < df['low'].iloc[i - 20] and df['low'].iloc[i] < df['low'].iloc[i + 20]
        right_shoulder = df['low'].iloc[i + 20] < df['low'].iloc[i + 10]
        if left_shoulder and head and right_shoulder:
            df.at[i, 'HeadShoulders'] = 1

    # TOBO (Ters Omuz Baş Omuz)
    df['InverseHeadShoulders'] = 0
    for i in range(20, len(df) - 20):
        left_shoulder = df['low'].iloc[i - 20] < df['low'].iloc[i - 10]
        head = df['low'].iloc[i] < df['low'].iloc[i - 20] and df['low'].iloc[i] < df['low'].iloc[i + 20]
        right_shoulder = df['low'].iloc[i + 20] < df['low'].iloc[i + 10]
        if left_shoulder and head and right_shoulder:
            df.at[i, 'InverseHeadShoulders'] = 1

    # Çanak Formasyonu
    df['Cup'] = 0
    df['Handle'] = 0
    min_depth = 10
    min_height = 5
    for i in range(min_depth, len(df) - min_depth):
        potential_cup = (df['low'].iloc[i - min_depth:i].min() == df['low'].iloc[i - min_depth:i].iloc[0])
        potential_cup &= (df['high'].iloc[i] > df['high'].iloc[i - min_depth - 1])
        if potential_cup:
            df.at[i, 'Cup'] = 1
            
        if df['Cup'].iloc[i - min_depth] == 1:
            handle_high = df['high'].iloc[i] < df['high'].iloc[i - min_depth] * 0.98
            handle_low = df['low'].iloc[i] > df['low'].iloc[i - min_depth] * 0.98
            if handle_high and handle_low:
                df.at[i, 'Handle'] = 1

    # Ters Çanak Formasyonu
    df['InverseCup'] = 0
    df['InverseHandle'] = 0
    for i in range(min_depth, len(df) - min_depth):
        potential_cup = (df['high'].iloc[i - min_depth:i].max() == df['high'].iloc[i - min_depth:i].iloc[0])
        potential_cup &= (df['low'].iloc[i] < df['low'].iloc[i - min_depth - 1])
        if potential_cup:
            df.at[i, 'InverseCup'] = 1
            
        if df['InverseCup'].iloc[i - min_depth] == 1:
            handle_high = df['low'].iloc[i] > df['low'].iloc[i - min_depth] * 0.98
            handle_low = df['high'].iloc[i] < df['high'].iloc[i - min_depth] * 0.98
            if handle_high and handle_low:
                df.at[i, 'InverseHandle'] = 1

    return df

def detect_rectangle_pattern(df, min_width=10):
    df['Rectangle'] = 0
    for i in range(min_width, len(df) - min_width):
        high_range = df['high'][i - min_width:i].max()
        low_range = df['low'][i - min_width:i].min()
        if abs(high_range - low_range) < 0.02 * (high_range + low_range):
            df.at[i, 'Rectangle'] = 1
    return df

def detect_flag_pennant_pattern(df):
    df['Flag'] = 0
    df['Pennant'] = 0
    for i in range(1, len(df) - 1):
        if (df['high'].iloc[i - 1] < df['high'].iloc[i] and df['low'].iloc[i] < df['low'].iloc[i - 1]) or (df['high'].iloc[i - 1] > df['high'].iloc[i] and df['low'].iloc[i] > df['low'].iloc[i - 1]):
            df.at[df.index[i], 'Flag'] = 1
        
        if (df['high'].iloc[i - 1] < df['high'].iloc[i] and df['high'].iloc[i] > df['high'].iloc[i + 1] and df['low'].iloc[i - 1] < df['low'].iloc[i] and df['low'].iloc[i] < df['low'].iloc[i + 1]):
            df.at[df.index[i], 'Pennant'] = 1
    return df

def detect_cone_patterns(df):
    df['RisingWedge'] = 0
    df['FallingWedge'] = 0
    for i in range(10, len(df) - 10):
        if df['high'].iloc[i - 10] < df['high'].iloc[i] and df['high'].iloc[i] < df['high'].iloc[i + 10] and df['low'].iloc[i - 10] > df['low'].iloc[i] and df['low'].iloc[i] > df['low'].iloc[i + 10]:
            df.at[df.index[i], 'RisingWedge'] = 1
        if df['high'].iloc[i - 10] > df['high'].iloc[i] and df['high'].iloc[i] > df['high'].iloc[i + 10] and df['low'].iloc[i - 10] < df['low'].iloc[i] and df['low'].iloc[i] < df['low'].iloc[i + 10]:
            df.at[df.index[i], 'FallingWedge'] = 1
    return df

# ML modeli eğitimi
def train_ml_model(df):
    df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)  # Sonraki mumda fiyatın artışına dayalı hedef
    features = ['EMA12', 'EMA26', 'RSI', 'MACD', 'UpperBand', 'LowerBand', 'ATR', 'Momentum', 'ADX', 'OBV', 'StochRSI_K', 'StochRSI_D',
                'DoubleTop', 'DoubleBottom', 'HeadShoulders', 'InverseHeadShoulders', 'Cup', 'Handle', 'InverseCup', 'InverseHandle',
                'Rectangle', 'Flag', 'Pennant', 'RisingWedge', 'FallingWedge']

    # Ensure all feature columns are numeric
    df[features] = df[features].apply(pd.to_numeric, errors='coerce')

    # Drop rows with any NaN values in the features columns
    df = df.dropna(subset=features)

    # Debug: Print DataFrame shape and head after dropping NaN values
    logging.info(f"DataFrame shape after dropping NaN values: {df.shape}")
    logging.info(df.head())

    if df.empty:
        logging.warning("DataFrame is empty after dropping NaN values. Cannot train model.")
        return None, None

    X = df[features]
    y = df['Target'].iloc[X.index]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model Hiperparametre Optimizasyonu
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    model = GradientBoostingClassifier(random_state=42)

    grid_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=50, cv=3, n_jobs=-1, random_state=42)
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_

    accuracy = model.score(X_test, y_test)
    logging.info(f"Makine öğrenimi modeli doğruluğu: {accuracy:.2f}")
    send_telegram_message(f"Makine öğrenimi modeli doğruluğu: {accuracy:.2f}")

    joblib.dump(model, 'ml_model.pkl')  # Modeli kaydet
    joblib.dump(scaler, 'scaler.pkl')  # Scaler'ı kaydet

    return model, scaler
# ML modeli ile tahmin
def predict_with_ml(model, scaler, df):
    features = ['EMA12', 'EMA26', 'RSI', 'MACD', 'UpperBand', 'LowerBand', 'ATR', 'Momentum', 'ADX', 'OBV', 'StochRSI_K', 'StochRSI_D',
                'DoubleTop', 'DoubleBottom', 'HeadShoulders', 'InverseHeadShoulders', 'Cup', 'Handle', 'InverseCup', 'InverseHandle',
                'Rectangle', 'Flag', 'Pennant', 'RisingWedge', 'FallingWedge']
    X_live = scaler.transform(df[features].iloc[[-1]])  # Son veriyi al
    prediction = model.predict(X_live)[0]
    return "LONG" if prediction == 1 else "SHORT"

# Zamanlı Stop-Loss ve Kademeli Stop-Loss fonksiyonları
def time_based_stop_loss(df, time_period=60):
    last_price = df['close'].iloc[-1]
    start_price = df['close'].iloc[-time_period]
    pct_change = (last_price - start_price) / start_price
    
    if pct_change < -0.02:  # %2'lik zarar durdurma seviyesi
        logging.info(f"Zamanlı Stop-Loss tetiklendi: {pct_change*100:.2f}% zarar")
        send_telegram_message(f"Zamanlı Stop-Loss tetiklendi: {pct_change*100:.2f}% zarar")
        return True
    return False

def trailing_stop_loss(df, atr_multiplier=1.5):
    atr = df['ATR'].iloc[-1]
    stop_loss_price = df['close'].iloc[-1] - (atr * atr_multiplier)
    return stop_loss_price

# Pozisyon yönetimi
def dynamic_risk_management(balance, atr, risk_percentage, prediction, max_drawdown=0.10):
    risk_amount = balance * risk_percentage
    stop_loss_distance = atr
    position_size = int(risk_amount / stop_loss_distance)
    
    # Volatilite bazlı risk yönetimi
    if atr > 0.05:  # Eğer ATR yüksekse, risk düzeyini azaltabiliriz.
        position_size = max(1, position_size // 2)  # Volatilite yüksekse risk azaltılır.

    # Max drawdown limit
    if position_size * stop_loss_distance > max_drawdown:
        position_size = int(max_drawdown / stop_loss_distance)
        return False, position_size
    
    return True, position_size

# Limit Order fonksiyonu
def place_order(symbol, order_type, quantity, stop_loss, take_profit):
    order = client.futures_create_order(
        symbol=symbol,
        side=order_type,
        type='MARKET' if not stop_loss else 'STOP_MARKET',
        quantity=quantity,
        stopPrice=stop_loss if stop_loss else take_profit,
        timeInForce='GTC'
    )
    logging.info(f"{order_type} order placed: {order}")
    send_telegram_message(f"{order_type} order placed for {symbol}: {order}")
    return order

# Hedef belirleme (Kademeli Stop-Loss ve Take-Profit)
def define_targets(df, atr_multiplier=2.0):
    last_close = df['close'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    stop_loss = last_close - (atr * atr_multiplier)
    take_profit = last_close + (atr * atr_multiplier)
    return stop_loss, take_profit

# Backtest fonksiyonu
def backtest(symbol, df, atr_multiplier=2.0, risk_percentage=0.01):
    initial_balance = 1000  # İlk deneme için başlangıç bakiyesi
    balance = initial_balance
    position_size = 0
    positions = []
    trades = 0
    
    for i in range(atr_multiplier, len(df)):
        last_price = df['close'].iloc[i]
        atr = df['ATR'].iloc[i]
        stop_loss, take_profit = define_targets(df.iloc[:i], atr_multiplier)
        
        order_type = predict_with_ml(model, scaler, df.iloc[i:i + 1])
        is_valid, position_size = dynamic_risk_management(balance, atr, risk_percentage, order_type, max_drawdown=0.10)
        
        if not is_valid:
            continue
        
        position = {
            'symbol': symbol,
            'entry_price': last_price,
            'entry_date': df.index[i],
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'quantity': position_size,
            'order_type': order_type
        }
        
        positions.append(position)
        
        # Simulate trade execution
        if order_type == "LONG":
            balance -= position['quantity'] * position['entry_price']
        elif order_type == "SHORT":
            balance += position['quantity'] * position['entry_price']
        
        trades += 1
        
        # Simulate stop loss or take profit hit
        if (order_type == "LONG" and last_price <= stop_loss) or (order_type == "SHORT" and last_price >= stop_loss) or ((order_type == "LONG" and last_price >= take_profit) or (order_type == "SHORT" and last_price <= take_profit)):
            trades -= 1
            if (order_type == "LONG" and last_price >= take_profit) or (order_type == "SHORT" and last_price <= stop_loss):
                balance += position['quantity'] * (take_profit if order_type == "LONG" else stop_loss)
            else:
                balance += position['quantity'] * last_price
            positions = [p for p in positions if p['entry_price'] != position['entry_price']]
            logging.info(f"Trade closed at {last_price}, new balance: {balance}")
            send_telegram_message(f"Trade closed at {last_price}, new balance: {balance}")
    
    logging.info(f"Total Trades: {trades}, Final Balance: {balance}")
    send_telegram_message(f"Total Trades: {trades}, Final Balance: {balance}")
    return balance, trades




# Ana işlev
def main():
    valid_symbols = [s['symbol'] for s in client.futures_exchange_info()['symbols']]
    
    for symbol in symbols:
        if symbol not in valid_symbols:
            logging.warning(f"Symbol {symbol} is not valid on Binance Futures. Skipping...")
            continue
        
        logging.info(f"Processing symbol: {symbol}")
        try:
            historical_klines = client.futures_klines(symbol=symbol, interval=timeframe, limit=limit)
        except BinanceAPIException as e:
            logging.error(f"Error fetching klines for symbol {symbol}: {e}")
            continue
        
        df = pd.DataFrame(historical_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        if len(df) < 21:
            logging.warning(f"Not enough data for symbol {symbol}. Skipping...")
            continue
        
        df = calculate_indicators(df)
        df = detect_candlestick_patterns(df)
        df = detect_rectangle_pattern(df)
        df = detect_flag_pennant_pattern(df)
        df = detect_cone_patterns(df)
        
        logging.info(f"DataFrame shape: {df.shape}")
        logging.info(df.head())
        
        if df.empty:
            logging.warning(f"DataFrame is empty after processing for symbol {symbol}. Skipping...")
            continue
        
        features = ['EMA12', 'EMA26', 'RSI', 'MACD', 'UpperBand', 'LowerBand', 'ATR', 'Momentum', 'ADX', 'OBV', 'StochRSI_K', 'StochRSI_D',
                    'DoubleTop', 'DoubleBottom', 'HeadShoulders', 'InverseHeadShoulders', 'Cup', 'Handle', 'InverseCup', 'InverseHandle',
                    'Rectangle', 'Flag', 'Pennant', 'RisingWedge', 'FallingWedge']
        
        df[features] = df[features].apply(pd.to_numeric, errors='coerce')
        df.fillna(0, inplace=True)  # Fill NaN values instead of dropping
        
        if df.empty:
            logging.warning(f"DataFrame is empty after filling NaN values for symbol {symbol}. Skipping...")
            continue
        
        model, scaler = train_ml_model(df)
        
        if model is not None and scaler is not None:
            backtest(symbol, df)


if __name__ == "__main__":
    main()
