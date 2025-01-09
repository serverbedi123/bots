import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import os
import glob

# Model ve scaler dosyalarını kaydetmek için dizin oluşturma
os.makedirs("models", exist_ok=True)

# Veri dizini
data_dir = "data"

def load_and_prepare_data():
    """Tüm sembol verilerini yükle ve birleştir"""
    all_data = []
    
    # data dizinindeki tüm CSV dosyalarını bul
    csv_files = glob.glob(os.path.join(data_dir, "*_data.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir} directory")
    
    print(f"Found {len(csv_files)} CSV files")
    
    for file in csv_files:
        try:
            # Sembol adını dosya adından çıkar
            symbol = os.path.basename(file).replace("_data.csv", "")
            
            # CSV dosyasını oku
            df = pd.read_csv(file)
            
            # Sembol sütunu ekle
            df['symbol'] = symbol
            
            all_data.append(df)
            print(f"Loaded data for {symbol}")
            
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
            continue
    
    if not all_data:
        raise ValueError("No data could be loaded from CSV files")
    
    # Tüm verileri birleştir
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Tarihe göre sırala
    combined_data['timestamp'] = pd.to_datetime(combined_data['timestamp'])
    combined_data = combined_data.sort_values('timestamp')
    
    return combined_data

try:
    # Verileri yükle
    print("Loading data...")
    data = load_and_prepare_data()
    
    # Özellikleri hazırla
    print("Preparing features...")
    features = data[['open', 'high', 'low', 'close', 'volume']]
    
    # Etiketleri oluştur (bir sonraki kapanış fiyatı yükselirse 1, düşerse 0)
    labels = (data.groupby('symbol')['close'].shift(-1) > data['close']).astype(int)
    labels = labels[:-1]  # Son satırı kaldır çünkü next_close değeri yok
    features = features[:-1]  # Etiketlerle eşleşmesi için son satırı kaldır
    
    # Veriyi ölçekle
    print("Scaling features...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)
    
    # Eğitim ve test setlerine ayır
    print("Splitting data...")
    train_size = int(len(scaled_features) * 0.8)
    X_train = scaled_features[:train_size]
    X_test = scaled_features[train_size:]
    y_train = labels[:train_size]
    y_test = labels[train_size:]
    
    # Modeli oluştur ve eğit
    print("Training model...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Model performansını değerlendir
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Testing accuracy: {test_score:.4f}")
    
    # Modeli ve scaler'ı kaydet
    print("Saving model and scaler...")
    joblib.dump(model, 'models/ml_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("Model and scaler successfully saved.")

except Exception as e:
    print(f"Error during training: {str(e)}")