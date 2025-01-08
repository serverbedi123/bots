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
            logging.info("Calculating basic technical indicators...")
        
            # RSI hesaplama
            df['RSI'] = ta.rsi(df['close'], length=14)
        
            # MACD hesaplama
            macd_data = ta.macd(df['close'])
            df['MACD'] = macd_data['MACD_12_26_9']
            df['MACD_SIGNAL'] = macd_data['MACDs_12_26_9']
            df['MACD_HIST'] = macd_data['MACDh_12_26_9']
        
            # Bollinger Bands hesaplama
            bollinger = ta.bbands(df['close'], length=20, std=2)
            df['BB_UPPER'] = bollinger['BBU_20_2.0']
            df['BB_MIDDLE'] = bollinger['BBM_20_2.0']
            df['BB_LOWER'] = bollinger['BBL_20_2.0']
        
            # Moving Averages
            df['SMA_20'] = ta.sma(df['close'], length=20)
            df['EMA_20'] = ta.ema(df['close'], length=20)
        
            # NaN değerleri temizle
            df = df.ffill().bfill()
        
            # Hesaplanan göstergeleri kontrol et
            required_indicators = ['RSI', 'MACD', 'MACD_SIGNAL', 'BB_UPPER', 'BB_LOWER']
            missing_indicators = [ind for ind in required_indicators if ind not in df.columns]
        
            if missing_indicators:
                logging.warning(f"Missing indicators after calculation: {missing_indicators}")
            else:
                logging.info("All required indicators calculated successfully")
            
            return df
        
        except Exception as e:
            logging.error(f"İndikatör hesaplama hatası: {str(e)}")
            return df

    def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """İleri seviye indikatörleri hesapla"""
        try:
            # DataFrame kontrolü
            if df.empty:
                logging.error("DataFrame is empty. Cannot calculate advanced indicators.")
                return df

            # Ichimoku hesaplaması
            try:
                ichimoku = ta.ichimoku(df['high'], df['low'], df['close'])
            
                # Ichimoku bileşenlerini ayrı ayrı ekle
                if isinstance(ichimoku, pd.DataFrame):
                    column_mapping = {
                    'ITS_9': 'ICHIMOKU_CONVERSION',
                    'IKS_26': 'ICHIMOKU_BASE',
                    'ISA_26': 'ICHIMOKU_SPAN_A',
                    'ISB_52': 'ICHIMOKU_SPAN_B',
                    'ICS_26': 'ICHIMOKU_CHIKOU'
                    }
                
                    for old_col, new_col in column_mapping.items():
                        if old_col in ichimoku.columns:
                            df[new_col] = ichimoku[old_col]
                        
                logging.info("Ichimoku indicators calculated successfully")
            
            except Exception as ichimoku_error:
                logging.error(f"Ichimoku calculation error: {ichimoku_error}")

            # ADX hesaplaması
            try:
                adx = ta.adx(df['high'], df['low'], df['close'])
                if isinstance(adx, pd.DataFrame):
                    if 'ADX_14' in adx.columns:
                        df['ADX'] = adx['ADX_14']
                    elif 'ADX' in adx.columns:
                     df['ADX'] = adx['ADX']
                logging.info("ADX calculated successfully")
            
            except Exception as adx_error:
                logging.error(f"ADX calculation error: {adx_error}")

            # NaN değerleri temizle - Update this part
            df = df.ffill().bfill()  # Using the recommended methods instead of fillna
        
            return df

        except Exception as e:
            logging.error(f"İleri seviye indikatör hesaplama hatası: {str(e)}")
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
        # Özellik isimlerini belirterek DataFrame oluştur
            feature_names = ['open', 'high', 'low', 'close', 'volume']
            features = df[feature_names].iloc[-1].to_frame().T
        
            # Ölçeklendirme işlemi
            scaled_features = self.scaler.transform(features)
        
            # Tahmin
            prediction = self.model.predict(scaled_features)
            probability = self.model.predict_proba(scaled_features)[0][prediction[0]]
        
            return {
            'type': 'BUY' if prediction[0] == 1 else 'SELL',
            'probability': probability
        }
        except Exception as e:
            logging.error(f"ML sinyal üretim hatası: {e}")
        return {'type': 'NONE', 'probability': 0.0}

    def generate_signals(self, df: pd.DataFrame) -> dict:
        """Teknik analiz sinyalleri üret"""
        try:
            required_columns = ['RSI', 'MACD', 'MACD_SIGNAL', 'BB_UPPER', 'BB_LOWER']
        
            # Gerekli sütunların varlığını kontrol et
            missing_columns = [col for col in required_columns if col not in df.columns]
            if df.empty or missing_columns:
                logging.warning(f"Missing columns for signal generation: {missing_columns}")
                return {'type': 'NONE', 'reason': 'missing_data'}

            last_row = df.iloc[-1]
            signals = []

            # RSI Sinyali
            if 'RSI' in df.columns:
                rsi = last_row['RSI']
            if rsi < 30:
                signals.append('BUY')
            elif rsi > 70:
                signals.append('SELL')

        # MACD Sinyali
            if all(col in df.columns for col in ['MACD', 'MACD_SIGNAL']):
                if last_row['MACD'] > last_row['MACD_SIGNAL']:
                    signals.append('BUY')
                else:
                    signals.append('SELL')

            # Bollinger Bands Sinyali
            if all(col in df.columns for col in ['BB_UPPER', 'BB_LOWER']):
                if last_row['close'] < last_row['BB_LOWER']:
                 signals.append('BUY')
                elif last_row['close'] > last_row['BB_UPPER']:
                    signals.append('SELL')

            # Ichimoku Sinyali
            ichimoku_columns = ['ICHIMOKU_CONVERSION', 'ICHIMOKU_BASE']
            if all(col in df.columns for col in ichimoku_columns):
                if last_row['ICHIMOKU_CONVERSION'] > last_row['ICHIMOKU_BASE']:
                    signals.append('BUY')
                else:
                    signals.append('SELL')

            # Sinyal kararı
            if signals:
                buy_signals = signals.count('BUY')
                sell_signals = signals.count('SELL')
            
            if buy_signals > sell_signals:
                return {'type': 'BUY', 'strength': buy_signals / len(signals)}
            elif sell_signals > buy_signals:
                return {'type': 'SELL', 'strength': sell_signals / len(signals)}

            return {'type': 'HOLD', 'strength': 0}

        except Exception as e:
            logging.error(f"Signal generation error: {str(e)}")
            return {'type': 'NONE', 'reason': 'error'}
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

    def get_symbol_info(self, symbol: str):
        """Sembol bilgilerini al"""
        try:
            # Sync versiyonu kullan
            exchange_info = self.client.exchange_info()
            for s in exchange_info['symbols']:
                if s['symbol'] == symbol:
                    return {
                        'pricePrecision': s['pricePrecision'],
                        'quantityPrecision': s['quantityPrecision']
                    }
            return None
        except Exception as e:
            logging.error(f"Sembol bilgisi alma hatası: {e}")
            return None

    def round_to_precision(self, value: float, precision: int) -> float:
        """Değeri belirtilen hassasiyete yuvarla"""
        factor = 10 ** precision
        return float(round(value * factor) / factor)

    async def execute_trade_with_risk_management(self, symbol: str, signal_type: str, current_price: float):
        """İşlem yönetimi ve risk kontrolü"""
        try:
            # Sembol bilgilerini al
            symbol_info = self.get_symbol_info(symbol)  # await kaldırıldı
            if not symbol_info:
                logging.error(f"Sembol bilgisi alınamadı: {symbol}")
                return False

            # Hesap bakiyesini al
            balance = self.get_account_balance()  # Eğer async değilse await kaldırıldı
        
            # Risk hesaplama (örnek: bakiyenin %1'i)
            risk_amount = balance * self.config.get('risk_percentage', 0.01)
        
            # Pozisyon büyüklüğünü hesapla ve yuvarla
            quantity = risk_amount / current_price
            quantity = self.round_to_precision(quantity, symbol_info['quantityPrecision'])
        
            # İşlem fiyatını yuvarla
            price = self.round_to_precision(current_price, symbol_info['pricePrecision'])

            try:
                if signal_type == 'BUY':
                    order = self.client.new_order(
                        symbol=symbol,
                        side='BUY',
                        type='MARKET',
                        quantity=quantity
                    )
                    logging.info(f"Long pozisyon açıldı: {symbol}, Miktar: {quantity}")
                    await self.send_telegram(f"🟢 Long Pozisyon Açıldı\nSymbol: {symbol}\nMiktar: {quantity}\nFiyat: {price}")
                
                elif signal_type == 'SELL':
                    order = self.client.new_order(
                        symbol=symbol,
                        side='SELL',
                        type='MARKET',
                        quantity=quantity
                    )
                    logging.info(f"Short pozisyon açıldı: {symbol}, Miktar: {quantity}")
                    await self.send_telegram(f"🔴 Short Pozisyon Açıldı\nSymbol: {symbol}\nMiktar: {quantity}\nFiyat: {price}")
                
                return True
            
            except Exception as order_error:
                logging.error(f"Order yerleştirme hatası: {order_error}")
                await self.send_telegram(f"⚠️ İşlem Hatası: {symbol} - {str(order_error)}")
                return False
            
        except Exception as e:
            logging.error(f"İşlem yönetimi hatası: {e}")
            await self.send_telegram(f"⚠️ İşlem Yönetimi Hatası: {symbol} - {str(e)}")
            return False
        


    def get_account_balance(self):
        """Hesap bakiyesini al"""
        try:
            account = self.client.balance()
            for asset in account:
                if asset['asset'] == 'USDT':
                    return float(asset['balance'])
            return 0.0
        except Exception as e:
            logging.error(f"Bakiye alma hatası: {e}")
            return 0.0
          
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
        try:
            logging.info(f"Bot started by {self.config.get('created_by', 'unknown')}")
            await self.send_telegram("🚀 Trading Bot Activated")
        
            while True:
                try:
                    # Trading saatleri kontrolü
                    if self.is_trading_allowed():
                        for symbol in self.config['symbols']:
                            # Mum verilerini al
                            df = self.get_klines(symbol)
                            if df.empty:
                                logging.warning(f"No data received for {symbol}")
                                continue

                            # Temel göstergeleri hesapla
                            df = self.calculate_indicators(df)
                            logging.info(f"Basic indicators calculated for {symbol}")

                            # İleri seviye göstergeleri hesapla
                            df = self.calculate_advanced_indicators(df)
                            logging.info(f"Advanced indicators calculated for {symbol}")

                         # ML ve teknik sinyalleri üret
                            ml_signal = self.generate_ml_signals(df)
                            technical_signal = self.generate_signals(df)

                            # Sinyalleri doğrula
                            if self._validate_signals(ml_signal, technical_signal):
                                current_price = float(df['close'].iloc[-1])
                                await self.execute_trade_with_risk_management(
                                    symbol, ml_signal, current_price
                                )

                            # Rate limit kontrolü
                            await asyncio.sleep(self.rate_limit_delay)

                    # Günlük istatistikleri sıfırla
                    if datetime.now().date() > self.last_daily_reset:
                        self.reset_daily_stats()

                    # Ana döngü bekleme süresi
                    await asyncio.sleep(self.config['check_interval'])

                except Exception as loop_error:
                    logging.error(f"Loop iteration error: {loop_error}")
                    await self.send_telegram(f"⚠️ Error in main loop: {loop_error}")
                    await asyncio.sleep(60)  # Hata durumunda 1 dakika bekle

        except Exception as e:
            logging.error(f"Critical error in run method: {e}")
            await self.send_telegram("🚨 Bot stopped due to critical error!")
            raise

if __name__ == "__main__":
    # Logging ayarları
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='advanced_trading_bot.log'
    )

    try:
        # Bot instance'ını oluştur
        bot = BinanceFuturesBot()
        
        # Modern asyncio kullanımı
        asyncio.run(bot.run())
        
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")
    except Exception as e:
        logging.error(f"Critical error: {e}")
