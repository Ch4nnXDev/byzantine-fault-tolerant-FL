
from datasets.dataLoader import DataManager
from models.cnn import CNN
from torchvision import datasets
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


class DependancyService:

    def __init__(self):
        pass

    def create_loader(self):

        return DataManager(
            datasets.MNIST
        ).load_dataset()

    def create_model(self):

        return CNN()

    def create_optimiser(self, model, learning_rate=0.01):
        return Adam(
            model.parameters(),
            lr=learning_rate
        )

    def create_loss_function(self):

        return CrossEntropyLoss()

