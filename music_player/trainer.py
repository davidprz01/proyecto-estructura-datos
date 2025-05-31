import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier                     
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from audio import FMAFeatureExtractor

# Configuración optimizada
BASE_PATH = os.path.join("fma", "data", "fma_small")
GENRES = ['000', '001', '002', '003', '004']  # Hasta 004
SAMPLES_PER_GENRE = 80  # 80 canciones por género (480 total)
TEST_SIZE = 0.2  # 20% para prueba

def load_dataset():
    # Carga el dataset con manejo robusto de errores
    extractor = FMAFeatureExtractor()
    X, y = [], []
    
    print("Cargando dataset (géneros 000-004)...")
    for genre_id in GENRES:
        genre_path = os.path.join(BASE_PATH, genre_id)
        if not os.path.exists(genre_path):
            print(f" Advertencia: {genre_path} no existe")
            continue
            
        print(f"\n Procesando género {genre_id}:")
        count = 0
        for file in os.listdir(genre_path):
            if count >= SAMPLES_PER_GENRE:
                break
                
            if file.endswith(('.mp3', '.wav')):
                full_path = os.path.join(genre_path, file)
                features = extractor.extract(full_path)
                
                if features is not None:
                    X.append(features)
                    y.append(int(genre_id))  # 000 → 0, 001 → 1, etc.
                    count += 1
                    if count % 20 == 0:
                        print(f"{count} canciones procesadas")
        
        print(f" {count} canciones válidas de {genre_id}")
    
    return np.array(X), np.array(y)

def train_model(X, y):
    # Pipeline de entrenamiento optimizado
    print("\n Entrenando modelo...")
    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=80,  # Balance velocidad-precisión
            max_depth=12,
            class_weight='balanced',
            n_jobs=-1,  # Usa todos los núcleos
            random_state=42
        )
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    
    model.fit(X_train, y_train)
    
    # Evaluación detallada
    print("\n Resultados:")
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print("\nReporte por género:")
    print(classification_report(y_test, y_pred, target_names=GENRES))
    
    return model

def main():
    try:
        X, y = load_dataset()
        
        if len(X) == 0:
            raise ValueError(" No se encontraron archivos válidos")
            
        print(f"\n Dataset cargado: {len(X)} muestras")
        print(f" Distribución: {np.bincount(y)}")
        
        model = train_model(X, y)
        
        # Guardar modelo
        model_path = "fma_model_6genres.pkl"
        joblib.dump(model, model_path)
        print(f"\n Modelo guardado como '{model_path}'")
        
        # Verificación
        loaded_model = joblib.load(model_path)
        print(f" Modelo verificado. Espera {loaded_model.n_features_in_} características")
        
    except Exception as e:
        print(f"\nError crítico: {str(e)}")

if __name__ == "__main__":
    main()
