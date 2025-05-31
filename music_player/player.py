import pygame
import tkinter as tk
from tkinter import filedialog
import os
from main import Queue
from classifier import FMAGenreClassifier

class MusicPlayer:
    def __init__(self):
        pygame.mixer.init()
        self.queue = Queue()
        self.classifier = FMAGenreClassifier()
        self.current_song = None
        self.is_playing = False
        self.is_paused = False  # Nueva variable para manejar el estado de pausa


        # Configuración básica de la ventana
        self.root = tk.Tk()
        self.root.title("Reproductor")
        self.root.geometry("300x400")
        self.root.configure(bg="#121212")
        
        # Marco principal
        self.main_frame = tk.Frame(self.root, bg="#121212")
        self.main_frame.pack(pady=10, fill='both', expand=True)
        self.root.configure(bg="black")


        # Botón para añadir canciones
        self.add_button = tk.Button(
            self.main_frame,
            text="Añadir Canciones", fg = "white",
            command=self.add_song,
            bg="#1b1b1d"
        )
        self.add_button.pack(pady=5, fill='x')

        # Lista de reproducción
        self.playlist = tk.Listbox(
            self.main_frame,
            height=12,
            fg="white", bg="#1b1b1d",
            selectbackground="#252424",
            selectforeground="white"
        )
        self.playlist.pack(pady=5, fill='both', expand=True)

        # Información de la canción
        self.song_label = tk.Label(
            self.main_frame,
            text="No hay canción seleccionada",
            fg="white", bg="#121212",
            wraplength=280,
        )
        self.song_label.pack(pady=5)

        self.genre_label = tk.Label(
            self.main_frame,
            text="Género: Desconocido",
            fg="white", bg="#121212"
        )
        self.genre_label.pack()

        # Controles básicos
        self.controls_frame = tk.Frame(self.main_frame, bg="#121212")
        self.controls_frame.pack(pady=10)

        self.play_button = tk.Button(
            self.controls_frame,
            text="▶",
            command=self.play_pause,
            width=5,
            fg="white", bg="#1b1b1d"
        )
        self.play_button.pack(side='left', padx=5)

        self.next_button = tk.Button(
            self.controls_frame,
            text="⏭",
            command=self.next_song,
            width=5,
            fg="white", bg="#1b1b1d"
        )
        self.next_button.pack(side='left', padx=5)

        # Eventos
        # self.playlist.bind("<Double-Button-1>", self.play_selected)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def add_song(self):
        filepaths = filedialog.askopenfilenames(filetypes=[("Audio Files", "*.mp3 *.wav")])
        for filepath in filepaths:
            self.queue.enqueue(filepath)
            self.playlist.insert("end", os.path.basename(filepath))

    def play_pause(self):
        if self.is_paused:
            # Si está pausado, reanudar
            self.resume()
        elif self.is_playing:
            # Si está reproduciendo, pausar
            self.pause()
        else:
            # Si no está reproduciendo nada, reproducir nueva canción
            self.play_new_song()

    def play_new_song(self):
        """Reproduce una nueva canción de la cola"""
        if not self.queue.is_empty():
            self.current_song = self.queue.dequeue()
            self.playlist.delete(0)
            
            # Mostrar información
            song_name = os.path.basename(self.current_song)
            self.song_label.config(text=song_name)
            
            # Clasificar género
            genre, confidence = self.classifier.predict_genre(self.current_song)
            self.genre_label.config(text=f"Género: {genre} ({confidence*100:.0f}%)")
            
            pygame.mixer.music.load(self.current_song)
            pygame.mixer.music.play()
            self.is_playing = True
            self.is_paused = False
            self.play_button.config(text="⏸")

    def pause(self):
        """Pausa la reproducción actual"""
        pygame.mixer.music.pause()
        self.is_playing = False
        self.is_paused = True
        self.play_button.config(text="▶")

    def resume(self):
        """Reanuda la reproducción pausada"""
        pygame.mixer.music.unpause()
        self.is_playing = True
        self.is_paused = False
        self.play_button.config(text="⏸")

    def next_song(self):
        """Pasa a la siguiente canción"""
        if self.is_playing or self.is_paused:
            pygame.mixer.music.stop()
        self.is_playing = False
        self.is_paused = False
        self.play_new_song()


    def close(self):
        pygame.mixer.quit()
        self.root.destroy()

if __name__ == "__main__":
    player = MusicPlayer()
    player.root.mainloop() 
