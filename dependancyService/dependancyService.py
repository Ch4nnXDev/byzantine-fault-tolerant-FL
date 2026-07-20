from datasets.dataLoader import DataManager
from models.cnn import CNN
from torchvision import datasets
from torch.nn import CrossEntropyLoss

from torch.optim import Adam

class DependancyService: ## The Dependancy Service Should foloow the factory pattern because the FL clients need to independantly make the instaces and its not a Singleton process its multiple.
    def __init__(self):
        pass
    
    def create_loader(self):
        return DataManager(datasets.MNIST).load_dataset()
        
    def create_model(self):
        return CNN()
    
    def create_optimiser(self, model):
        optimiser = Adam(
            model.parameters(),
            lr=0.01
        )
        return optimiser
    
    def create_loss_function(self):
        return CrossEntropyLoss() ##Cross Entropy Loss Does SoftMax No need of that transformation inside the model
        
        
        
        