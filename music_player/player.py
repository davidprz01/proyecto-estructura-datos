import pygame
import customtkinter as ctk
from genre import GenreClassifier
from audio import AudioFeatureExtractor
from song_queue import SongQueue

class MusicPlayer(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Reproductor con Clasificador de Género")
        self.geometry("800x500")
        
        # Inicializa pygame y la cola
        pygame.mixer.init()
        self.queue = SongQueue()
        
        # Carga el modelo ML
        self.feature_extractor = AudioFeatureExtractor()
        self.genre_classifier = GenreClassifier()
        self.genre_classifier.load_model()  # Asegúrate de tener "genre_model.pkl"
        
        # Interfaz
        self.setup_ui()
    
    def setup_ui(self):
        """Configura la interfaz gráfica."""
        # Botón para agregar canciones
        self.add_button = ctk.CTkButton(
            self, text="Agregar Canción", command=self.add_song
        )
        self.add_button.pack(pady=10)
        
        # Lista de reproducción
        self.playlist_box = tk.Listbox(
            self, bg="#2b2b2b", fg="white", font=("Arial", 12)
        )
        self.playlist_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Etiqueta para mostrar el género
        self.genre_label = ctk.CTkLabel(self, text="Género: Desconocido")
        self.genre_label.pack(pady=5)
        
        # Botón de reproducción
        self.play_button = ctk.CTkButton(
            self, text="Reproducir", command=self.play_song
        )
        self.play_button.pack(pady=5)
    
    def add_song(self):
        """Añade una canción a la cola."""
        filepath = filedialog.askopenfilename(filetypes=[("MP3 Files", "*.mp3")])
        if filepath:
            self.queue.enqueue(filepath)
            self.playlist_box.insert(tk.END, os.path.basename(filepath))
    
    def play_song(self):
        """Reproduce la canción actual y muestra su género."""
        if not self.queue.is_empty():
            current_song = self.queue.peek()
            pygame.mixer.music.load(current_song)
            pygame.mixer.music.play()
            
            # Predice el género
            features = self.feature_extractor.extract_features(current_song)
            genre, confidence = self.genre_classifier.predict(features)
            
            # Actualiza la UI
            self.genre_label.configure(
                text=f"Género: {genre} (Confianza: {confidence:.0%})"
            )