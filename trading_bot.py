import pandas as pd
import numpy as np
import json
import joblib
from binance.client import Client
from binance.exceptions import BinanceAPIException  # Remove APIError
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import pandas_ta as ta
import asyncio
from telegram import Bot
import time
import sys  # Add this import
# Config dosyasını yükle
# Config dosyasını yükle
with open("config.json") as f:
    config = json.load(f)

# Binance API bilgileri
api_key = config["api_key"]
# RSA API için api_secret gerekmez
use_testnet = config.get("use_testnet", True)

# Telegram API bilgileri
telegram_bot_token = config["telegram_bot_token"]
chat_id = config["telegram_chat_id"]

# Binance bağlantısı (RSA için)
client = Client(
    api_key=api_key,
    api_secret=None,  # RSA için None olmalı
    testnet=use_testnet,
    rsa_key=True  # RSA kullanımını belirt
)

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parametreler
symbols = config["symbols"]
leverage = config["leverage"]
risk_per_trade = config["risk_per_trade"]
timeframe = config["timeframe"]
limit = config["limit"]

# Telegram bildirim fonksiyonu
async def send_telegram_message(message):
    try:
        bot = Bot(token=telegram_bot_token)
        async with bot:
            await bot.send_message(chat_id=chat_id, text=message)
        logging.info(f"Telegram message sent: {message}")
    except Exception as e:
        logging.error(f"Telegram message error: {str(e)}")
        
        
def calculate_indicators(df):
    try:
        # Convert data types and clean NaN values
        for col in ['close', 'high', 'low', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Use ffill() and bfill() instead of fillna(method='ffill')
        df = df.ffill().bfill()
        
        # Calculate indicators
        df['EMA12'] = ta.ema(df['close'], length=12)
        df['EMA26'] = ta.ema(df['close'], length=26)
        df['RSI'] = ta.rsi(df['close'], length=14)
        
        # Fix MACD calculation
        macd = ta.macd(df['close'])
        df['MACD'] = macd['MACD_12_26_9']
        df['MACDSignal'] = macd['MACDs_12_26_9']  # Changed from MACDSb to MACDs
        df['MACDHist'] = macd['MACDh_12_26_9']
        
        # Calculate other indicators
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
        
        # Final cleanup using new methods
        df = df.ffill().fillna(0)
        
        # Verify all required columns exist
        required_columns = ['UpperBand', 'LowerBand', 'ATR', 'Momentum', 
                          'ADX', 'OBV', 'StochRSI_K', 'StochRSI_D']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"Missing columns: {missing_columns}")
            
        return df
        
    except Exception as e:
        logging.error(f"calculate_indicators error: {str(e)}")
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
    try:
        # Add more features for better prediction
        df['Price_Change'] = df['close'].pct_change()
        df['Volume_Change'] = df['volume'].pct_change()
        df['High_Low_Range'] = (df['high'] - df['low']) / df['low']
        df['Close_Open_Range'] = (df['close'] - df['open']) / df['open']
        
        # Add rolling means
        df['Price_MA5'] = df['close'].rolling(window=5).mean()
        df['Price_MA20'] = df['close'].rolling(window=20).mean()
        df['Volume_MA5'] = df['volume'].rolling(window=5).mean()
        
        # Add momentum indicators
        df['ROC'] = (df['close'] / df['close'].shift(10) - 1) * 100
        df['MFI'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'])
        
        # Create target with more sophisticated logic
        df['Target'] = ((df['close'].shift(-1) > df['close']) & 
                       (df['volume'] > df['volume'].rolling(window=20).mean()) &
                       (df['RSI'] < 70) & (df['RSI'] > 30)).astype(int)

        features = [
            'EMA12', 'EMA26', 'RSI', 'MACD', 'UpperBand', 'LowerBand', 'ATR', 
            'Momentum', 'ADX', 'OBV', 'StochRSI_K', 'StochRSI_D',
            'Price_Change', 'Volume_Change', 'High_Low_Range', 'Close_Open_Range',
            'Price_MA5', 'Price_MA20', 'Volume_MA5', 'ROC', 'MFI',
            'DoubleTop', 'DoubleBottom', 'HeadShoulders', 'InverseHeadShoulders',
            'Cup', 'Handle', 'InverseCup', 'InverseHandle',
            'Rectangle', 'Flag', 'Pennant', 'RisingWedge', 'FallingWedge'
        ]

        # Handle NaN values more carefully
        df[features] = df[features].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=features + ['Target'])

        if df.empty or len(df) < 100:  # Ensure enough data
            logging.warning("Not enough data for training")
            return None, None

        X = df[features]
        y = df['Target']

        # Split with proper time series consideration
        train_size = int(len(df) * 0.8)
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Improved hyperparameter grid
        param_grid = {
            'n_estimators': [200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4],
            'subsample': [0.8, 0.9, 1.0]
        }

        model = GradientBoostingClassifier(random_state=42)
        
        # Cross validation with time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        grid_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=20,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )

        grid_search.fit(X_train_scaled, y_train)
        model = grid_search.best_estimator_

        # Evaluate model
        train_accuracy = model.score(X_train_scaled, y_train)
        test_accuracy = model.score(X_test_scaled, y_test)
        
        logging.info(f"Training accuracy: {train_accuracy:.2f}")
        logging.info(f"Test accuracy: {test_accuracy:.2f}")

        # Only save model if accuracy is above threshold
        if test_accuracy > 0.55:  # Minimum threshold for useful predictions
            joblib.dump(model, 'ml_model.pkl')
            joblib.dump(scaler, 'scaler.pkl')
            return model, scaler
        else:
            logging.warning("Model accuracy too low, not saving model")
            return None, None

    except Exception as e:
        logging.error(f"train_ml_model error: {str(e)}")
        return None, None
# ML modeli ile tahmin
def predict_with_ml(model, scaler, df):
    try:
        features = ['EMA12', 'EMA26', 'RSI', 'MACD', 'UpperBand', 'LowerBand', 'ATR', 'Momentum', 
                    'ADX', 'OBV', 'StochRSI_K', 'StochRSI_D',
                    'DoubleTop', 'DoubleBottom', 'HeadShoulders', 'InverseHeadShoulders', 
                    'Cup', 'Handle', 'InverseCup', 'InverseHandle',
                    'Rectangle', 'Flag', 'Pennant', 'RisingWedge', 'FallingWedge']

        # Fill NaN values with the mean of the respective columns
        df[features] = df[features].fillna(df[features].mean())

        # Check for NaN values in features
        if df[features].isnull().sum().sum() > 0:
            logging.error("NaN values found in features, skipping prediction.")
            return None

        # En son satırdan tahmin yapmak
        X_live = scaler.transform(df[features].iloc[[-1]])
        prediction = model.predict(X_live)[0]
        return "LONG" if prediction == 1 else "SHORT"

    except Exception as e:
        logging.error(f"predict_with_ml error: {str(e)}")
        return None
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




def verify_api_permissions():
    try:
        # Test API key permissions
        logging.info("Verifying API permissions...")
        
        # Check if API key exists and has correct format
        if not api_key:
            logging.error("API key is missing in config.json")
            return False
        if not api_secret:
            logging.error("API secret is missing in config.json")
            return False
            
        # Log API key length for debugging (without revealing the key)
        logging.info(f"API key length: {len(api_key)}")
        logging.info(f"API secret length: {len(api_secret)}")
        
        # Initialize test client with testnet first
        test_client = Client(api_key, api_secret, testnet=True)
        logging.info("Test client initialized")
        
        try:
            # Test basic connectivity
            test_client.ping()
            logging.info("Successfully pinged Binance API")
            
            # Test futures API access
            futures_account = test_client.futures_account()
            logging.info("Successfully accessed futures account")
            
            # Get account info for permissions check
            account_info = test_client.get_account()
            permissions = account_info.get('permissions', [])
            logging.info(f"Account permissions: {permissions}")
            
            # Check for required permissions
            required_permissions = ['SPOT', 'FUTURES']
            missing_permissions = [p for p in required_permissions if p not in permissions]
            
            if missing_permissions:
                logging.error(f"Missing required permissions: {', '.join(missing_permissions)}")
                logging.error("Please enable these permissions in your Binance account settings:")
                for p in missing_permissions:
                    logging.error(f"- {p}")
                return False
            
            # Test futures-specific endpoints
            try:
                # Test futures data access
                test_client.futures_exchange_info()
                logging.info("Successfully accessed futures exchange info")
                
                # Test futures account balance
                balance = test_client.futures_account_balance()
                logging.info(f"Futures account balance verified: {len(balance)} currencies found")
                
            except BinanceAPIException as e:
                logging.error(f"Futures-specific API test failed: {e}")
                return False
                
            logging.info("All API permissions verified successfully")
            return True
            
        except BinanceAPIException as e:
            logging.error(f"API test failed during connectivity check: {e}")
            return False
            
    except BinanceAPIException as e:
        if e.code == -2015:
            logging.error("Invalid API key, IP, or permissions. Please check:")
            logging.error("1. API key and secret are correct in config.json")
            logging.error("2. IP restriction is disabled or your IP is whitelisted")
            logging.error("3. Futures trading is enabled for your account")
            logging.error("4. API key has Futures trading permission")
            logging.error(f"Full error: {str(e)}")
        else:
            logging.error(f"Binance API error code {e.code}: {str(e)}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error verifying API permissions: {str(e)}")
        logging.error(f"Error type: {type(e).__name__}")
        return False
    
def initialize_binance_client():
    try:
        logging.info("Initializing Binance client with RSA authentication...")
        
        # RSA API client oluştur
        client = Client(
            api_key=api_key,
            api_secret=None,  # RSA için None
            testnet=use_testnet,
            rsa_key=True
        )
        logging.info("✓ RSA API client initialized")
        
        # Bağlantıyı test et
        try:
            client.ping()
            logging.info("✓ Connection test successful")
        except BinanceAPIException as e:
            logging.error(f"Connection test failed: {e}")
            return None
            
        # Futures API'yi test et
        try:
            futures_account = client.futures_account()
            logging.info("✓ Futures account access successful")
        except BinanceAPIException as e:
            logging.error(f"Futures account access failed: {e}")
            return None
            
        # Position mode'u ayarla
        try:
            client.futures_change_position_mode(dualSidePosition=False)
            logging.info("✓ Position mode set to one-way")
        except BinanceAPIException as e:
            if e.code != -4059:  # Zaten one-way modda
                logging.error(f"Failed to set position mode: {e}")
                return None
            logging.info("✓ Already in one-way position mode")
            
        # Test sembolü ile marjin tipini ayarla
        test_symbol = "BTCUSDT"
        try:
            client.futures_change_margin_type(
                symbol=test_symbol,
                marginType='ISOLATED'
            )
            logging.info(f"✓ Set {test_symbol} to ISOLATED margin")
        except BinanceAPIException as e:
            if e.code != -4046:  # Zaten ISOLATED
                logging.error(f"Failed to set margin type: {e}")
                return None
            logging.info(f"✓ {test_symbol} already in ISOLATED margin")
            
        # Sembolleri yapılandır
        for symbol in symbols:
            try:
                # Marjin tipi
                try:
                    client.futures_change_margin_type(
                        symbol=symbol,
                        marginType='ISOLATED'
                    )
                except BinanceAPIException as e:
                    if e.code != -4046:
                        continue
                        
                # Kaldıraç
                client.futures_change_leverage(
                    symbol=symbol,
                    leverage=leverage
                )
                logging.info(f"✓ Configured {symbol}: ISOLATED margin, {leverage}x leverage")
                
            except BinanceAPIException as e:
                logging.warning(f"Could not configure {symbol}: {e}")
                continue
                
            time.sleep(0.1)
            
        return client
        
    except BinanceAPIException as e:
        logging.error(f"Failed to initialize RSA client: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return None
def verify_futures_account():
    try:
        test_client = Client(api_key, api_secret)
        
        # Check futures account status
        futures_account = test_client.futures_account()
        logging.info(f"Futures account status: Active")
        logging.info(f"Available balance: {futures_account['totalWalletBalance']} USDT")
        
        # Check if futures trading is enabled
        if float(futures_account['totalWalletBalance']) <= 0:
            logging.warning("Futures wallet balance is 0. Please transfer funds to your futures wallet")
        
        return True
    except BinanceAPIException as e:
        if e.code == -2015:
            logging.error("Futures account not activated. Please activate futures trading in your Binance account")
        else:
            logging.error(f"Futures account verification failed: {e}")
        return False 
    
def verify_futures_permissions():
    logging.info("Verifying futures trading permissions...")
    
    try:
        test_client = Client(api_key, api_secret)
        
        # Step 1: Check if futures account exists
        try:
            futures_account = test_client.futures_account()
            logging.info("✓ Futures account accessed successfully")
        except BinanceAPIException as e:
            logging.error("✗ Could not access futures account. Please enable futures trading on Binance")
            logging.error(f"Error: {e}")
            return False

        # Step 2: Check if USDT-M futures is enabled
        try:
            test_client.futures_get_position_mode()
            logging.info("✓ USDT-M Futures enabled")
        except BinanceAPIException as e:
            logging.error("✗ USDT-M Futures not enabled")
            logging.error("Please enable USDT-M futures trading in your Binance account")
            return False

        # Step 3: Verify margin type access
        try:
            test_client.futures_change_margin_type(symbol='BTCUSDT', marginType='ISOLATED')
            logging.info("✓ Margin type modification permitted")
        except BinanceAPIException as e:
            if e.code != -4046:  # Already set to ISOLATED
                logging.error("✗ Cannot modify margin type")
                return False
            else:
                logging.info("✓ Margin type already set to ISOLATED")

        # Step 4: Verify leverage modification
        try:
            test_client.futures_change_leverage(symbol='BTCUSDT', leverage=5)
            logging.info("✓ Leverage modification permitted")
        except BinanceAPIException as e:
            logging.error("✗ Cannot modify leverage")
            return False

        logging.info("All futures permissions verified successfully")
        return True

    except BinanceAPIException as e:
        logging.error(f"Futures verification failed: {e}")
        if e.code == -2015:
            logging.error("\nTo fix this:")
            logging.error("1. Go to Binance.com -> API Management")
            logging.error("2. Edit your API key")
            logging.error("3. Enable 'Enable Futures' permission")
            logging.error("4. Temporarily disable IP restrictions")
            logging.error("5. Save changes and wait 5 minutes")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during futures verification: {e}")
        return False      
def validate_trading_symbols():
    try:
        test_client = Client(api_key, api_secret)
        logging.info("Validating trading symbols...")
        
        # Get all valid futures symbols
        exchange_info = test_client.futures_exchange_info()
        valid_symbols = [s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING']
        
        # Check if configured symbols are valid
        invalid_symbols = [symbol for symbol in symbols if symbol not in valid_symbols]
        
        if invalid_symbols:
            logging.error(f"Invalid symbols found: {', '.join(invalid_symbols)}")
            logging.info("Valid USDT-M futures symbols include: BTCUSDT, ETHUSDT, BNBUSDT, etc.")
            return False
            
        logging.info(f"Validated symbols: {', '.join(symbols)}")
        return True
        
    except BinanceAPIException as e:
        logging.error(f"Error validating symbols: {e}")
        return False  
    

    
    
def validate_futures_symbols():
    try:
        test_client = Client(api_key, api_secret)
        logging.info("Validating futures trading symbols...")
        
        # Get futures exchange info
        exchange_info = test_client.futures_exchange_info()
        valid_futures_symbols = {s['symbol']: s for s in exchange_info['symbols'] if s['status'] == 'TRADING'}
        
        # Validate each configured symbol
        valid_symbols = []
        invalid_symbols = []
        
        for symbol in symbols:
            if symbol in valid_futures_symbols:
                symbol_info = valid_futures_symbols[symbol]
                # Store symbol precision and filters for later use
                valid_symbols.append({
                    'symbol': symbol,
                    'pricePrecision': symbol_info['pricePrecision'],
                    'quantityPrecision': symbol_info['quantityPrecision'],
                    'minQty': float(symbol_info['filters'][1]['minQty']),
                    'maxQty': float(symbol_info['filters'][1]['maxQty']),
                    'stepSize': float(symbol_info['filters'][1]['stepSize'])
                })
            else:
                invalid_symbols.append(symbol)
        
        if invalid_symbols:
            logging.error(f"Invalid futures symbols found: {', '.join(invalid_symbols)}")
            return False, None
        
        logging.info(f"Successfully validated {len(valid_symbols)} symbols")
        return True, valid_symbols
        
    except BinanceAPIException as e:
        logging.error(f"Error validating futures symbols: {e}")
        return False, None
    except Exception as e:
        logging.error(f"Unexpected error during symbol validation: {e}")
        return False, None    
def load_config():
    try:
        with open("config.json") as f:
            config = json.load(f)
            
        # RSA API kontrolü
        if not config.get('api_key'):
            logging.error("RSA API key is missing in config.json")
            return None
            
        # API key format kontrolü
        if len(config['api_key']) != 64:
            logging.error("Invalid RSA API key format")
            return None
            
        # Config validasyonu
        required_fields = [
            'api_key', 'symbols', 'leverage', 
            'risk_per_trade', 'timeframe', 'limit',
            'telegram_bot_token', 'telegram_chat_id'
        ]
        
        for field in required_fields:
            if field not in config:
                logging.error(f"Missing required field in config.json: {field}")
                return None
                
        # Sembolleri kontrol et
        if not isinstance(config['symbols'], list):
            logging.error("'symbols' must be a list in config.json")
            return None
            
        # Sembolleri büyük harfe çevir
        config['symbols'] = [s.upper() for s in config['symbols']]
        
        return config
        
    except FileNotFoundError:
        logging.error("config.json not found")
        return None
    except json.JSONDecodeError:
        logging.error("Invalid JSON in config.json")
        return None
    except Exception as e:
        logging.error(f"Error loading config: {str(e)}")
        return None
def monitor_symbol_status():
    try:
        test_client = Client(api_key, api_secret)
        
        # Get current futures prices
        prices = test_client.futures_symbol_ticker()
        price_dict = {p['symbol']: float(p['price']) for p in prices}
        
        # Get 24h stats
        stats = test_client.futures_ticker()
        stats_dict = {s['symbol']: s for s in stats}
        
        logging.info("\nSymbol Status Report:")
        for symbol in symbols:
            if symbol in price_dict:
                stat = stats_dict.get(symbol, {})
                logging.info(f"{symbol}:")
                logging.info(f"  Price: {price_dict[symbol]}")
                logging.info(f"  24h Volume: {stat.get('volume', 'N/A')}")
                logging.info(f"  24h Change: {stat.get('priceChangePercent', 'N/A')}%")
            else:
                logging.warning(f"{symbol}: Not available for trading")
                
    except Exception as e:
        logging.error(f"Error monitoring symbols: {e}")   
def update_available_symbols():
    try:
        client = Client(api_key, api_secret)
        exchange_info = client.futures_exchange_info()
        available_symbols = [s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING']
        
        # Filter configured symbols
        valid_symbols = [s for s in symbols if s in available_symbols]
        
        if len(valid_symbols) != len(symbols):
            removed_symbols = set(symbols) - set(valid_symbols)
            logging.warning(f"Some symbols are no longer available: {', '.join(removed_symbols)}")
            
        return valid_symbols
    except Exception as e:
        logging.error(f"Error updating available symbols: {e}")
        return []

def monitor_trading_status():
    while True:
        try:
            valid_symbols = update_available_symbols()
            if not valid_symbols:
                logging.error("No valid symbols available for trading")
                continue
                
            # Get current prices and 24h stats
            prices = client.futures_symbol_ticker()
            price_dict = {p['symbol']: float(p['price']) for p in prices}
            
            # Log active symbols status
            for symbol in valid_symbols:
                if symbol in price_dict:
                    logging.info(f"{symbol}: {price_dict[symbol]}")
                    
            time.sleep(300)  # Check every 5 minutes
            
        except Exception as e:
            logging.error(f"Error monitoring trading status: {e}")
            time.sleep(60)    

def validate_api_key():
    logging.info("Validating API key configuration...")
    try:
        test_client = Client(api_key, api_secret, testnet=True)
        
        # Test basic connectivity
        test_client.ping()
        logging.info("✓ Basic connectivity test passed")
        
        # Get API key permissions
        account = test_client.get_account()
        
        # Check permissions
        permissions = account.get('permissions', [])
        logging.info(f"API Key permissions: {permissions}")
        
        # Required permissions
        required = ['SPOT', 'FUTURES']
        missing = [p for p in required if p not in permissions]
        
        if missing:
            logging.error(f"Missing required permissions: {', '.join(missing)}")
            logging.error("\nTo fix this:")
            logging.error("1. Go to Binance.com")
            logging.error("2. Navigate to API Management")
            logging.error("3. Delete your current API key")
            logging.error("4. Create a new API key with these permissions:")
            logging.error("   - Enable Reading")
            logging.error("   - Enable Spot & Margin Trading")
            logging.error("   - Enable Futures")
            logging.error("   - Enable Universal Transfer")
            logging.error("5. Temporarily disable IP restriction")
            return False
            
        logging.info("✓ API key permissions validated")
        return True
        
    except BinanceAPIException as e:
        logging.error(f"API validation failed: {e}")
        if e.code == -2015:
            logging.error("\nTo fix Invalid API-key error:")
            logging.error("1. Create a new API key on Binance.com")
            logging.error("2. Enable ALL permissions for futures trading")
            logging.error("3. Disable IP restrictions temporarily")
            logging.error("4. Update config.json with new API credentials")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during API validation: {e}")
        return False         
# Ana işlev
def main():
    
    while True:
        try:
            for symbol in symbols:
                logging.info(f"Processing symbol: {symbol}")
                
                # Mum verilerini al
                historical_klines = client.futures_klines(symbol=symbol, interval=timeframe, limit=limit)
                
                # DataFrame oluştur
                df = pd.DataFrame(historical_klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                # Veri tiplerini dönüştür
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['timestamp'] = df['timestamp'].apply(lambda x: int(x.timestamp()))
                df.set_index('timestamp', inplace=True)
                
                # Ensure DataFrame has enough rows
                if len(df) < 1:
                    logging.warning(f"DataFrame for {symbol} is empty or does not have enough rows.")
                    continue
                
                # İndikatörleri hesapla
                df = calculate_indicators(df)
                
                # Ensure DataFrame has enough rows after calculating indicators
                if len(df) < 1:
                    logging.warning(f"DataFrame for {symbol} is empty or does not have enough rows after calculating indicators.")
                    continue
                
                # Formasyonları tespit et
                df = detect_candlestick_patterns(df)
                df = detect_rectangle_pattern(df)
                df = detect_flag_pennant_pattern(df)
                df = detect_cone_patterns(df)
                
                if not df.empty:
                    model, scaler = train_ml_model(df)
                    
                    if model is not None and scaler is not None:
                        # Son veriye göre tahmin yap
                        if len(df) < 1:
                            logging.warning(f"DataFrame for {symbol} does not have enough rows for prediction.")
                            continue
                        
                        prediction = predict_with_ml(model, scaler, df)
                        
                        # İşlem sinyali varsa
                        if prediction in ["LONG", "SHORT"]:
                            # Risk yönetimi
                            try:
                                balance = float(client.futures_account_balance()[0]['balance'])
                                if len(df) < 1:
                                    logging.warning(f"DataFrame for {symbol} does not have enough rows for ATR calculation.")
                                    continue
                                
                                atr = df['ATR'].iloc[-1]
                                is_valid, position_size = dynamic_risk_management(
                                    balance, atr, risk_per_trade, prediction
                                )
                                
                                if is_valid:
                                    # Stop loss ve take profit belirle
                                    stop_loss, take_profit = define_targets(df)
                                    
                                    # Kaldıraç ayarla
                                    client.futures_change_leverage(
                                        symbol=symbol,
                                        leverage=leverage
                                    )
                                    
                                    # İşlem aç
                                    order = place_order(
                                        symbol, prediction, position_size,
                                        stop_loss, take_profit
                                    )
                                    
                                    asyncio.run(send_telegram_message(
                                        f"New {prediction} order placed for {symbol}\n"
                                        f"Entry: {df['close'].iloc[-1]}\n"
                                        f"Stop Loss: {stop_loss}\n"
                                        f"Take Profit: {take_profit}\n"
                                        f"Position Size: {position_size}"
                                    ))
                                    
                            except Exception as e:
                                logging.error(f"Trading error for {symbol}: {str(e)}")
                                continue
                
                # Her symbol arasında kısa bekleme
                time.sleep(1)
            
            # Her döngü sonunda bekleme
            time.sleep(60)
            
        except Exception as e:
            logging.error(f"Main loop error: {str(e)}")
            time.sleep(60)
            
if __name__ == "__main__":
    try:
        logging.info("Starting trading bot...")
        
        # Load and validate config
        config = load_config()
        if config is None:
            logging.error("Failed to load configuration. Please check config.json")
            sys.exit(1)
            
        # Update global variables from config
        api_key = config['api_key']
        api_secret = config['api_secret']
        symbols = config['symbols']
        leverage = config['leverage']
        
        # Initialize Binance client
        client = initialize_binance_client()
        if client is None:
            sys.exit(1)
            
        # Start main trading loop
        main()
        
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Critical error: {e}")
        sys.exit(1)
        
