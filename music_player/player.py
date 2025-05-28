import pygame
import os
from tkinter import Tk, Button, Label, filedialog, Listbox, Scrollbar, messagebox
from song_queue import SongQueue
from classifier import FMAGenreClassifier

class MusicPlayer:
    def __init__(self):
        pygame.mixer.init()
        self.queue = SongQueue()
        self.classifier = FMAGenreClassifier()
        self.current_song = None
        self.is_playing = False
        
        # Configuración de la interfaz
        self.root = Tk()
        self.root.title("Reproductor Musical con Clasificación de Género")
        self.root.geometry("500x400")
        
        # Elementos de la UI
        self.label = Label(self.root, text="Género: Desconocido", font=("Arial", 12))
        self.label.pack(pady=10)
        
        self.confidence_label = Label(self.root, text="Confianza: 0%", font=("Arial", 10))
        self.confidence_label.pack()
        
        # Lista de reproducción
        self.playlist = Listbox(self.root, selectmode="SINGLE", width=60, height=15)
        self.scrollbar = Scrollbar(self.root, orient="vertical")
        self.playlist.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.playlist.yview)
        self.playlist.pack(pady=10)
        self.scrollbar.pack(side="right", fill="y")
        
        # Botones
        Button(self.root, text="Añadir Canción", command=self.add_song).pack(side="left", padx=5)
        Button(self.root, text="Reproducir", command=self.play).pack(side="left", padx=5)
        Button(self.root, text="Pausa", command=self.pause).pack(side="left", padx=5)
        Button(self.root, text="Detener", command=self.stop).pack(side="left", padx=5)
        Button(self.root, text="Siguiente", command=self.next_song).pack(side="left", padx=5)
        
        # Eventos
        self.playlist.bind("<Double-Button-1>", self.play_selected)
        pygame.mixer.music.set_endevent(pygame.USEREVENT)
        
        self.root.mainloop()
    
    def add_song(self):
        filepaths = filedialog.askopenfilenames(filetypes=[("Audio Files", "*.mp3 *.wav")])
        for filepath in filepaths:
            self.queue.enqueue(filepath)
            self.playlist.insert("end", os.path.basename(filepath))
    
    def play(self):
        if not self.is_playing and not self.queue.is_empty():
            self.current_song = self.queue.dequeue()
            self.playlist.delete(0)  # Elimina la primera canción de la lista
            
            # Clasificar género
            genre, confidence = self.classifier.predict_genre(self.current_song)
            self.label.config(text=f"Género: {genre.capitalize()}")
            self.confidence_label.config(text=f"Confianza: {confidence*100:.1f}%")
            
            pygame.mixer.music.load(self.current_song)
            pygame.mixer.music.play()
            self.is_playing = True
    
    def play_selected(self, event):
        selection = self.playlist.curselection()
        if selection:
            index = selection[0]
            song_name = self.playlist.get(index)
            
            # Buscar la canción en la cola (simplificado para ejemplo)
            # En una implementación real, necesitarías mapear nombres a rutas
            self.current_song = song_name  # Esto debería ser la ruta completa
            self.play()
    
    def pause(self):
        if self.is_playing:
            pygame.mixer.music.pause()
            self.is_playing = False
        else:
            pygame.mixer.music.unpause()
            self.is_playing = True
    
    def stop(self):
        pygame.mixer.music.stop()
        self.is_playing = False
    
    def next_song(self):
        self.stop()
        self.play()
    
    def check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.USEREVENT:  # Canción terminó
                self.next_song()
        self.root.after(100, self.check_events)