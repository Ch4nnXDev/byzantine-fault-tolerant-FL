from dependancyService.dependancyService import DependancyService

class Server:
    

    def __init__(self):
        
        dependancy = DependancyService()
        self.model = dependancy.create_model()
        
        self.client_weight = []
        
        
    
    def send_model(self):
        
        pass
    
    def receive_weights():
        pass
    
    def aggregate(self):
        pass
    