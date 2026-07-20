import uuid
from dependancyService.dependancyService import DependancyService

class Client:
    def __init__(self):
        dependancy = DependancyService()
        self.id = uuid.uuid4()
        self.model = dependancy.create_model()
        self.train_loader, self.test_loader = dependancy.create_loader()
        self.criterion = dependancy.create_loss_function()
        self.optimiser = dependancy.create_optimiser(self.model)
        
    def train_model(self, epoch):
        
        self.model.train()
        
        
        
            
        
        