from typing import Optional

class SongNode:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.next = None

class SongQueue:
    def __init__(self):
        self.head = None
        self.tail = None
        self.length = 0
    
    def enqueue(self, filepath: str) -> None:
        """Añade una canción al final de la cola."""
        new_node = SongNode(filepath)
        if self.tail is None:
            self.head = self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node
        self.length += 1
    
    def dequeue(self) -> Optional[str]:
        """Elimina y devuelve la primera canción de la cola."""
        if self.head is None:
            return None
            
        removed = self.head
        self.head = self.head.next
        
        if self.head is None:
            self.tail = None
        
        self.length -= 1
        return removed.filepath
    
    def peek(self) -> Optional[str]:
        """Devuelve la primera canción de la cola sin eliminarla."""
        return self.head.filepath if self.head else None
    
    def is_empty(self) -> bool:
        """Comprueba si la cola está vacía."""
        return self.head is None
    
    def size(self) -> int:
        """Devuelve el número de canciones en la cola."""
        return self.length
    
    def clear(self) -> None:
        """Vacía la cola por completo."""
        self.head = self.tail = None
        self.length = 0