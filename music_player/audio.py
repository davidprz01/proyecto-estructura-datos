import librosa
import numpy as np
import warnings
from librosa.util.exceptions import ParameterError

class FMAFeatureExtractor:
    def __init__(self):
        warnings.filterwarnings("ignore", category=UserWarning)
    
    def extract(self, filepath, duration=30):
        try:
            # 1. Carga del audio con manejo de errores
            try:
                y, sr = librosa.load(filepath, sr=22050, mono=True, duration=duration)
            except Exception as e:
                print(f"Error al cargar {filepath}: {str(e)}")
                return None
                
            # 2. Verificación del audio
            if len(y) == 0 or np.max(np.abs(y)) < 0.001:
                print(f"Audio vacío/silencioso en {filepath}")
                return None

            # 3. Extracción robusta de características
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # MFCCs (13 coeficientes)
                mfcc = librosa.feature.mfcc(
                    y=y,
                    sr=sr,
                    n_mfcc=13,
                    n_fft=2048,
                    hop_length=512
                )
                mfcc_mean = np.mean(mfcc, axis=1)
                
                # Cálculo compatible del tempo para todas las versiones
                try:
                    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                    if hasattr(librosa.feature, 'rhythm'):  # Para librosa >= 0.10.0
                        tempo = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr)[0]
                    else:  # Para versiones antiguas
                        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
                except Exception as e:
                    print(f"Error calculando tempo en {filepath}: {str(e)}")
                    tempo = 120.0  # Valor por defecto
                
            # 4. Construcción del vector de características (14 dimensiones)
            features = np.concatenate([
                mfcc_mean,  # 13 coeficientes
                [tempo]     # + tempo = 14 características
            ])
            
            # Validación final
            if features.shape[0] != 14 or np.any(np.isnan(features)):
                raise ValueError("Características inválidas")
                
            return features
            
        except Exception as e:
            print(f"Error procesando {filepath}: {str(e)}")
            return None

# import librosa
# import numpy as np
# from sklearn.preprocessing import StandardScaler

# class FMAFeatureExtractor:
#     def __init__(self):
#         # Eliminamos el StandardScaler ya que debe ser parte del pipeline del modelo
#         pass
    
#     def extract(self, filepath, duration=30):
#         try:
#             y, sr = librosa.load(filepath, duration=duration, mono=True, sr=22050)
            
#             # 1. Solo características básicas para coincidir con las 14 esperadas
#             mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#             mfcc_mean = np.mean(mfcc, axis=1)
            
#             # 2. Solo tempo como característica adicional (total 14)
#             try:
#                 # Forma nueva (librosa ≥ 0.10.0)
#                 tempo = librosa.feature.rhythm.tempo(y=y, sr=sr)[0]
#             except AttributeError:
#                 # Forma vieja (librosa < 0.10.0)
#                 tempo = librosa.beat.tempo(y=y, sr=sr)[0]
            
#             features = np.concatenate([
#                 mfcc_mean,  # 13 características
#                 [tempo]     # 1 característica (total 14)
#             ])
            
#             if features.shape[0] != 14:
#                 raise ValueError(f"Se obtuvieron {features.shape[0]} características, se esperaban 14")
                
#             if np.any(np.isnan(features)):
#                 raise ValueError("Características contienen valores NaN")
                
#             return features
            
#         except Exception as e:
#             print(f"Error al procesar {filepath}: {str(e)}")
#             return None