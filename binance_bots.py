import asyncio
import json
import logging
from datetime import datetime, time
import time as time_module
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import pandas_ta as ta
from binance.um_futures import UMFutures
from binance.error import ClientError
from telegram import Bot
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib

class BinanceFuturesBot:
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.client = UMFutures(
            key=self.config['api_key'],
            secret=self.config['api_secret']
        )
        self.telegram = Bot(token=self.config['telegram_token'])
        self.chat_id = self.config['telegram_chat_id']
        self.positions = {}
        self.last_api_call = 0
        self.rate_limit_delay = 0.1
        self.model = self._load_ml_model()
        self.scaler = self._load_scaler()
        self.daily_trades = 0
        self.daily_stats = {
            'trades': 0,
            'profit': 0.0,
            'losses': 0.0
        }
        self.last_daily_reset = datetime.now().date()

    def _load_config(self, config_path: str) -> dict:
        """Config dosyasını yükle"""
        try:
            with open(config_path, 'r') as file:
                config = json.load(file)
            self._validate_config(config)
            return config
        except Exception as e:
            logging.error(f"Config yükleme hatası: {e}")
            raise

    def _validate_config(self, config: dict) -> None:
        """Config dosyasını doğrula"""
        required_fields = [
            'api_key', 'api_secret', 'telegram_token', 'telegram_chat_id',
            'symbols', 'risk_management', 'trading_hours', 'timeframes',
            'ml_model_path', 'scaler_path'
        ]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Eksik config alanı: {field}")

    def _load_ml_model(self) -> GradientBoostingClassifier:
        """Makine öğrenimi modelini yükle"""
        try:
            model = joblib.load(self.config['ml_model_path'])
            return model
        except Exception as e:
            logging.error(f"ML model yükleme hatası: {e}")
            raise

    def _load_scaler(self) -> StandardScaler:
        """Ölçekleyiciyi yükle"""
        try:
            scaler = joblib.load(self.config['scaler_path'])
            return scaler
        except Exception as e:
            logging.error(f"Scaler yükleme hatası: {e}")
            raise

    async def send_telegram(self, message: str) -> None:
        """Telegram mesajı gönder"""
        if self.config['notifications']['trade_updates']:
            try:
                await self.telegram.send_message(
                    chat_id=self.chat_id,
                    text=message
                )
            except Exception as e:
                logging.error(f"Telegram mesaj hatası: {e}")

    def get_klines(self, symbol: str) -> pd.DataFrame:
        """Mum verilerini al"""
        try:
            timeframe = self.config['timeframes']['default']
            klines = self.client.klines(
                symbol=symbol,
                interval=timeframe,
                limit=100
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades_count', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            return df
            
        except Exception as e:
            logging.error(f"Kline veri alma hatası: {e}")
            return pd.DataFrame()

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Temel teknik indikatörleri hesapla"""
        try:
            # Moving Averages
            df['SMA_20'] = ta.sma(df['close'], length=20)
            df['EMA_20'] = ta.ema(df['close'], length=20)
            
            # Bollinger Bands
            bb = ta.bbands(df['close'], length=20)
            df['BB_UPPER'] = bb['BBU_20_2.0']
            df['BB_MIDDLE'] = bb['BBM_20_2.0']
            df['BB_LOWER'] = bb['BBL_20_2.0']
            
            # RSI
            df['RSI'] = ta.rsi(df['close'], length=14)
            
            # MACD
            macd = ta.macd(df['close'])
            df['MACD'] = macd['MACD_12_26_9']
            df['MACD_SIGNAL'] = macd['MACDs_12_26_9']
            
            return df
            
        except Exception as e:
            logging.error(f"İndikatör hesaplama hatası: {e}")
            return df

    def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """İleri seviye indikatörleri hesapla"""
        try:
            # Ichimoku
            ichimoku = ta.ichimoku(df['high'], df['low'], df['close'])
            df['ICHIMOKU_BASE'] = ichimoku['ITS_9']
            df['ICHIMOKU_CONVERSION'] = ichimoku['IKS_26']
            
            # ADX
            adx = ta.adx(df['high'], df['low'], df['close'])
            df['ADX'] = adx['ADX_14']
            
            # Fibonacci Retracement Levels (example calculation)
            df['FIB_LEVEL_1'] = df['close'] + (df['high'] - df['low']) * 0.382
            df['FIB_LEVEL_2'] = df['close'] + (df['high'] - df['low']) * 0.618
            
            return df
            
        except Exception as e:
            logging.error(f"İleri seviye indikatör hesaplama hatası: {e}")
            return df

    def _calculate_atr(self, symbol: str) -> float:
        """ATR hesapla"""
        try:
            df = self.get_klines(symbol)
            atr = ta.atr(df['high'], df['low'], df['close'], length=14)
            return atr.iloc[-1]
        except Exception as e:
            logging.error(f"ATR hesaplama hatası: {e}")
            return 0.0

    def _calculate_dynamic_stop_loss(self, price: float, atr: float, trade_type: str, multiplier: float) -> float:
        """Dinamik stop loss hesapla"""
        if trade_type == 'BUY':
            return price - (atr * multiplier)
        elif trade_type == 'SELL':
            return price + (atr * multiplier)

    def _calculate_dynamic_take_profit(self, price: float, atr: float, trade_type: str, multiplier: float) -> float:
        """Dinamik take profit hesapla"""
        if trade_type == 'BUY':
            return price + (atr * multiplier)
        elif trade_type == 'SELL':
            return price - (atr * multiplier)

    async def _place_orders(self, symbol: str, trade_type: str, position_size: float, stop_loss: float, take_profit: float):
        """Order'ları yerleştir"""
        try:
            if trade_type == 'BUY':
                order = self.client.new_order(
                    symbol=symbol,
                    side='BUY',
                    type='MARKET',
                    quantity=position_size
                )
            elif trade_type == 'SELL':
                order = self.client.new_order(
                    symbol=symbol,
                    side='SELL',
                    type='MARKET',
                    quantity=position_size
                )
            # Add stop loss and take profit orders
            sl_order = self.client.new_order(
                symbol=symbol,
                side='SELL' if trade_type == 'BUY' else 'BUY',
                type='STOP_MARKET',
                stopPrice=stop_loss,
                quantity=position_size
            )
            tp_order = self.client.new_order(
                symbol=symbol,
                side='SELL' if trade_type == 'BUY' else 'BUY',
                type='TAKE_PROFIT_MARKET',
                stopPrice=take_profit,
                quantity=position_size
            )
            return order
        except Exception as e:
            logging.error(f"Order yerleştirme hatası: {e}")
            return None

    def generate_ml_signals(self, df: pd.DataFrame) -> dict:
        """ML sinyalleri üret"""
        try:
            features = df.iloc[-1][['open', 'high', 'low', 'close', 'volume']].values.reshape(1, -1)
            scaled_features = self.scaler.transform(features)
            prediction = self.model.predict(scaled_features)
            probability = self.model.predict_proba(scaled_features)[0][prediction[0]]
            return {'type': 'BUY' if prediction[0] == 1 else 'SELL', 'probability': probability}
        except Exception as e:
            logging.error(f"ML sinyal üretim hatası: {e}")
            return {'type': 'NONE', 'probability': 0.0}

    def generate_signals(self, df: pd.DataFrame) -> dict:
        """Teknik sinyalleri üret"""
        try:
            last_row = df.iloc[-1]
            if last_row['RSI'] > 70 and last_row['MACD'] < last_row['MACD_SIGNAL']:
                return {'type': 'SELL'}
            elif last_row['RSI'] < 30 and last_row['MACD'] > last_row['MACD_SIGNAL']:
                return {'type': 'BUY'}
            else:
                return {'type': 'HOLD'}
        except Exception as e:
            logging.error(f"Teknik sinyal üretim hatası: {e}")
            return {'type': 'NONE'}

    def _validate_signals(self, ml_signal: dict, technical_signal: dict) -> bool:
        """Sinyalleri doğrula"""
        try:
            if ml_signal['type'] == technical_signal['type'] and ml_signal['type'] != 'NONE':
                return True
            return False
        except Exception as e:
            logging.error(f"Sinyal doğrulama hatası: {e}")
            return False

    def is_trading_allowed(self) -> bool:
        """Trading koşullarını kontrol et"""
        current_hour = datetime.now().hour
        if not (self.config['trading_hours']['start_hour'] <= 
                current_hour < self.config['trading_hours']['end_hour']):
            return False
            
        if self.daily_trades >= self.config['risk_management']['max_trades_per_day']:
            return False
            
        return True

    def _calculate_position_size(self, balance: float, risk_per_trade: float) -> float:
        """Pozisyon büyüklüğünü hesapla"""
        return balance * risk_per_trade * self.config['max_position_size']

    async def execute_trade_with_risk_management(self, symbol: str, signal: dict, price: float):
        """Gelişmiş risk yönetimi ile trade gerçekleştirme"""
        try:
            if not self.is_trading_allowed():
                logging.info("Trading koşulları uygun değil")
                return

            if signal['type'] not in ['BUY', 'SELL']:
                return

            # Risk hesaplaması
            account = self.client.account()
            balance = float(account['totalWalletBalance'])
            position_size = self._calculate_position_size(
                balance,
                self.config['risk_management']['max_loss_percentage'] / 100
            )

            # Stop loss ve take profit
            atr = self._calculate_atr(symbol)
            stop_loss = self._calculate_dynamic_stop_loss(
                price, atr, signal['type'],
                self.config['risk_management']['stop_loss_multiplier']
            )
            take_profit = self._calculate_dynamic_take_profit(
                price, atr, signal['type'],
                self.config['risk_management']['take_profit_multiplier']
            )

            # Order'ları gönder
            order = await self._place_orders(
                symbol, signal['type'], position_size, stop_loss, take_profit
            )

            if order:
                self.daily_trades += 1
                await self._send_trade_notification(
                    symbol, signal, price, position_size, stop_loss, take_profit
                )

        except Exception as e:
            logging.error(f"Trade execution hatası: {e}")
            await self.send_telegram(f"Trade hatası: {e}")

    async def _send_trade_notification(self, symbol, signal, price, size, sl, tp):
        """Trade bildirimini gönder"""
        message = (
            f"🤖 Trade Executed\n"
            f"Symbol: {symbol}\n"
            f"Type: {signal['type']}\n"
            f"Price: {price:.8f}\n"
            f"Size: {size:.8f}\n"
            f"Stop Loss: {sl:.8f}\n"
            f"Take Profit: {tp:.8f}\n"
            f"Probability: {signal['probability']:.2f}"
        )
        await self.send_telegram(message)

    async def run(self):
        """Ana bot döngüsü"""
        logging.info(f"Bot started by {self.config.get('created_by', 'unknown')}")
        await self.send_telegram("🚀 Trading Bot Activated")

        while True:
            try:
                if self.is_trading_allowed():
                    for symbol in self.config['symbols']:
                        df = self.get_klines(symbol)
                        if df.empty:
                            continue

                        df = self.calculate_advanced_indicators(df)
                        ml_signal = self.generate_ml_signals(df)
                        technical_signal = self.generate_signals(df)

                        if self._validate_signals(ml_signal, technical_signal):
                            current_price = float(df['close'].iloc[-1])
                            await self.execute_trade_with_risk_management(
                                symbol, ml_signal, current_price
                            )

                        await asyncio.sleep(self.rate_limit_delay)

                await asyncio.sleep(self.config['check_interval'])

            except Exception as e:
                logging.error(f"Main loop error: {e}")
                await self.send_telegram(f"⚠️ Error: {e}")
                await asyncio.sleep(60)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='advanced_trading_bot.log'
    )

    try:
        bot = BinanceFuturesBot()
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")
    except Exception as e:
        logging.error(f"Critical error: {e}")
