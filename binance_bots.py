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
from sklearn.model_selection import GridSearchCV
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
        self.model = self._initialize_model()
        self.scaler = StandardScaler()
        self.daily_trades = 0
        self.daily_stats = {
            'trades': 0,
            'profit': 0.0,
            'losses': 0.0,
            'win_rate': 0.0
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
            'notifications', 'check_interval'
        ]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Eksik config alanı: {field}")

    def _initialize_model(self) -> GradientBoostingClassifier:
        """Initialize Gradient Boosting Model with hyperparameter tuning"""
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5]
        }
        model = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
        return grid_search

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

    def train_model(self, df: pd.DataFrame) -> None:
        """Train the ML model with new data."""
        try:
            features = ['SMA_20', 'EMA_20', 'RSI', 'MACD', 'BB_UPPER', 'BB_LOWER', 'ADX', 'STOCH', 'ICHIMOKU']
            X = df[features].dropna()
            y = (df['close'].shift(-1) > df['close']).astype(int).dropna()

            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
        except Exception as e:
            logging.error(f"Model eğitim hatası: {e}")

    def generate_ml_signals(self, df: pd.DataFrame) -> dict:
        """Generate signals based on ML model."""
        try:
            features = ['SMA_20', 'EMA_20', 'RSI', 'MACD', 'BB_UPPER', 'BB_LOWER', 'ADX', 'STOCH', 'ICHIMOKU']
            X = df[features].iloc[-1:].dropna()
            X_scaled = self.scaler.transform(X)

            probabilities = self.model.predict_proba(X_scaled)[0]
            signal_type = 'BUY' if probabilities[1] > 0.6 else 'SELL'
            return {
                'type': signal_type,
                'probability': probabilities[1] if signal_type == 'BUY' else probabilities[0]
            }
        except Exception as e:
            logging.error(f"ML sinyal üretim hatası: {e}")
            return {}

    def is_trading_allowed(self) -> bool:
        """Check if trading is allowed based on trading hours"""
        now = datetime.now().time()
        start_time = time(*map(int, self.config['trading_hours']['start'].split(':')))
        end_time = time(*map(int, self.config['trading_hours']['end'].split(':')))
        return start_time <= now <= end_time

    def _calculate_position_size(self, balance: float, risk: float, volatility: float) -> float:
        """Calculate position size based on risk management and volatility"""
        return balance * risk / volatility

    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate the Average True Range (ATR)"""
        return ta.atr(df['high'], df['low'], df['close'])[-1]

    def _calculate_dynamic_stop_loss(self, price: float, atr: float, signal_type: str, multiplier: float) -> float:
        """Calculate dynamic stop loss based on ATR"""
        return price - atr * multiplier if signal_type == 'BUY' else price + atr * multiplier

    def _calculate_dynamic_take_profit(self, price: float, atr: float, signal_type: str, multiplier: float) -> float:
        """Calculate dynamic take profit based on ATR"""
        return price + atr * multiplier if signal_type == 'BUY' else price - atr * multiplier

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
            df = self.get_klines(symbol)
            atr = self._calculate_atr(df)
            position_size = self._calculate_position_size(
                balance,
                self.config['risk_management']['max_loss_percentage'] / 100,
                atr
            )

            # Stop loss ve take profit
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

    def adjust_strategy_based_on_market_conditions(self, df: pd.DataFrame) -> None:
        """Adjust strategy parameters based on market conditions."""
        volatility = df['close'].pct_change().std()
        if volatility > self.config['risk_management']['volatility_threshold']:
            self.config['risk_management']['stop_loss_multiplier'] = 1.5
            self.config['risk_management']['take_profit_multiplier'] = 3
        else:
            self.config['risk_management']['stop_loss_multiplier'] = 2
            self.config['risk_management']['take_profit_multiplier'] = 4

    def log_performance(self):
        """Log daily performance statistics."""
        logging.info(f"Daily Trades: {self.daily_stats['trades']}")
        logging.info(f"Daily Profit: {self.daily_stats['profit']}")
        logging.info(f"Daily Losses: {self.daily_stats['losses']}")
        logging.info(f"Daily Win Rate: {self.daily_stats['win_rate']}")

    def handle_error(self, error: Exception) -> None:
        """Handle and log errors."""
        logging.error(f"Error: {error}")
        asyncio.run(self.send_telegram(f"⚠️ Error: {error}"))

    async def _place_orders(self, symbol: str, order_type: str, position_size: float, stop_loss: float, take_profit: float) -> dict:
        """Place orders with Binance API"""
        try:
            order = self.client.new_order(
                symbol=symbol,
                side='BUY' if order_type == 'BUY' else 'SELL',
                type='LIMIT',
                quantity=position_size,
                price=stop_loss if order_type == 'BUY' else take_profit,
                timeInForce='GTC'
            )
            return order
        except ClientError as e:
            logging.error(f"Order placement hatası: {e}")
            return {}

    async def _send_trade_notification(self, symbol: str, signal: dict, price: float, position_size: float, stop_loss: float, take_profit: float) -> None:
        """Send trade notification via Telegram"""
        message = (
            f"Trade Executed:\n"
            f"Symbol: {symbol}\n"
            f"Type: {signal['type']}\n"
            f"Price: {price}\n"
            f"Position Size: {position_size}\n"
            f"Stop Loss: {stop_loss}\n"
            f"Take Profit: {take_profit}\n"
            f"Probability: {signal['probability']}"
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

                        df = self.calculate_indicators(df)
                        self.train_model(df)
                        ml_signal = self.generate_ml_signals(df)

                        if ml_signal:
                            current_price = float(df['close'].iloc[-1])
                            await self.execute_trade_with_risk_management(
                                symbol, ml_signal, current_price
                            )

                        self.adjust_strategy_based_on_market_conditions(df)
                        self.log_performance()

                        await asyncio.sleep(self.rate_limit_delay)

                await asyncio.sleep(self.config['check_interval'])

            except Exception as e:
                self.handle_error(e)
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
