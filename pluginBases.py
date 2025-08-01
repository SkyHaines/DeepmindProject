from abc import ABC, abstractmethod

class BehaviourPlugin(ABC):
    @abstractmethod
    def add_parser_params(self, parser):
        pass
    
    @abstractmethod
    def run(self):
        pass
    
class GraphicsPlugin(ABC):
    @abstractmethod
    def draw(self, frame):
        pass