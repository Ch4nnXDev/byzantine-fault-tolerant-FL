from dependancyService.dependancyService import DependancyService
import torch

class Server:
    

    def __init__(self):
        
        dependancy = DependancyService()
        self.model = dependancy.create_model()
        
        self.received_weights = []
        
        
    def get_model_weights(self):
        return self.model.state_dict()
    
    def send_model(self, clients):
        
        weights = self.model.state_dict()
        for client in clients:
            client.set_weights(weights)
    
    def receive_weights(self, weights):
        
        self.received_weights.append(weights)
        
        
    
    def aggregate(self):
        new_weights = {}
        for key in self.received_weights[0].keys():
            new_weights[key] = torch.stack(
                [
                    weights[key].float()
                    for weights in self.client_weights
                    ]
                
            ).mean(dim=0)
        self.model.load_state_dict(
            new_weights
        )


        self.client_weights = []
    