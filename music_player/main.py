from typing import Optional

class Node:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.next = None

class Queue:
    def __init__(self):
        self.head = None
        self.last = None
        self.length = 0
    
    def enqueue(self, filepath: str) -> None:
        # Añade una canicon a la cola
        new_node = Node(filepath)
        if self.last is None:
            self.head = self.last = new_node
        else:
            self.last.next = new_node
            self.last = new_node
        self.length += 1
    
    def dequeue(self) -> Optional[str]:
        # Elimina y devuelve la canción de la cola.
        if self.head is None:
            return None
            
        removed = self.head
        self.head = self.head.next
        
        if self.head is None:
            self.last = None
        
        self.length -= 1
        return removed.filepath
    
    def peek(self) -> Optional[str]:
        # Devuelve la primera canción de la cola sin eliminarla
        return self.head.filepath if self.head else None
    
    def is_empty(self) -> bool:
        # Comprueba si la cola está vacía
        return self.head is None
    
    def size(self) -> int:
        # Devuelve el número de canciones en la cola
        return self.length
    
    def clear(self) -> None:
        # Vacía la cola por completo
        self.head = self.last = None
        self.length = 0