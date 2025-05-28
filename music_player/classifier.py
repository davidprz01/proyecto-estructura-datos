import joblib
import numpy as np
from audio import FMAFeatureExtractor

class FMAGenreClassifier:
    def __init__(self, model_path='fma_model_6genres.pkl'):
        """
        Clasificador optimizado para 6 géneros musicales (000-005 del FMA)
        
        Args:
            model_path (str): Ruta al modelo entrenado (.pkl)
        """
        try:
            # Cargar modelo (que incluye el StandardScaler en el pipeline)
            self.model = joblib.load(model_path)
            self.extractor = FMAFeatureExtractor()
            
            # Mapeo actualizado para 6 géneros
            self.genres = {
                0: "pop",         # 000
                1: "hiphop",      # 001
                2: "rock",        # 002
                3: "electronic",  # 003
                4: "experimental", # 004
                5: "folk"         # 005
            }
            
            print(f"✅ Modelo cargado correctamente")
            print(f"🔢 Géneros soportados: {len(self.genres)}")
            print(f"📏 Características esperadas: {self.model.n_features_in_}")
            
        except Exception as e:
            print(f"❌ Error al cargar el clasificador: {str(e)}")
            raise
    
    def predict_genre(self, audio_path):
        """
        Predice el género musical de un archivo de audio
        
        Args:
            audio_path (str): Ruta al archivo de audio
            
        Returns:
            tuple: (género_predicho, confianza) o ("unknown", 0.0) si falla
        """
        try:
            # 1. Extraer características
            features = self.extractor.extract(audio_path)
            if features is None:
                print(f"⚠️ No se pudieron extraer características de {audio_path}")
                return "unknown", 0.0
                
            # 2. Validar dimensionalidad
            if len(features) != self.model.n_features_in_:
                print(f"⚠️ Error dimensional: esperaba {self.model.n_features_in_} features, obtuve {len(features)}")
                return "unknown", 0.0
            
            # 3. Predecir (el pipeline aplica StandardScaler automáticamente)
            proba = self.model.predict_proba([features])[0]
            idx = np.argmax(proba)
            confidence = float(proba[idx])
            
            # 4. Devolver resultado
            return self.genres.get(idx, "unknown"), confidence
            
        except Exception as e:
            print(f"❌ Error durante la predicción: {str(e)}")
            return "unknown", 0.0

    def get_supported_genres(self):
        """Devuelve la lista de géneros soportados"""
        return list(self.genres.values())



# import joblib
# import numpy as np
# from audio import FMAFeatureExtractor

# class FMAGenreClassifier:
#     def __init__(self, model_path='fma_model.pkl'):
#         try:
#             # Cargar modelo (que debe incluir el scaler como parte del pipeline)
#             self.model = joblib.load(model_path)
#             self.extractor = FMAFeatureExtractor()
            
#             self.genres = {
#                 0: "pop", 1: "hiphop", 2: "rock", 
#                 3: "electronic", 4: "folk", 5: "jazz",
#                 6: "reggae", 7: "blues"
#             }
            
#             print(f"Modelo cargado. Espera {self.model.n_features_in_} características")
            
#         except Exception as e:
#             print(f"Error al cargar el clasificador: {str(e)}")
#             raise
    
#     def predict_genre(self, audio_path):
#         try:
#             features = self.extractor.extract(audio_path)
#             if features is None:
#                 return "unknown", 0.0
                
#             if len(features) != self.model.n_features_in_:
#                 raise ValueError(
#                     f"Discrepancia en características: "
#                     f"modelo espera {self.model.n_features_in_}, "
#                     f"obtenidas {len(features)}"
#                 )
            
#             # El modelo debe manejar la normalización internamente
#             prediction = self.model.predict_proba([features])[0]
#             idx = np.argmax(prediction)
            
#             return self.genres.get(idx, "unknown"), float(prediction[idx])
            
#         except Exception as e:
#             print(f"Error en predicción: {str(e)}")
#             return "unknown", 0.0