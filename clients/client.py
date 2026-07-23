import uuid
from dependancyService.dependancyService import DependancyService
import torch

class Client:
    def __init__(self):
        dependancy = DependancyService()
        self.id = uuid.uuid4()
        self.train_loader, self.test_loader = dependancy.create_loader()
        self.criterion = dependancy.create_loss_function()
        self.model = dependancy.create_model()
        self.optimiser = dependancy.create_optimiser(self.model)
        
    def train_model(self, epochs=2):
        
        self.model.train()
        
        for epoch in range(epochs):
            
            for images, labels in self.train_loader:
                
                outputs = self.model(images)
                
                loss = self.criterion(
                    outputs,
                    labels
                    
                    
                )
                
                self.optimiser.zero_grad()
                
                loss.backward()
                
                self.optimiser.step()
    
    
    def evaluate(self):
        
        self.model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                
                outputs = self.model(images)
                
                predicted = torch.argmax(
                    outputs,
                    dim=1
                )
                
                total += labels.size(0)
                
                correct += (predicted == labels).sum().item()
                
                
        return correct / total
    
    def get_weights(self):
        return self.model.state_dict()
    
    def set_weights(self, weights):
        self.model.load_state_dict(weights)
                    
                
                
        
        
        
            
        
        