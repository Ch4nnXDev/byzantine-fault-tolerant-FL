import uuid
from copy import deepcopy

from dependancyService.dependancyService import DependancyService


class Client:

    def __init__(self, train_loader):

        dependancy = DependancyService()

        self.id = uuid.uuid4()

        # This client receives only its own partition
        self.train_loader = train_loader

        self.criterion = dependancy.create_loss_function()
        self.model = dependancy.create_model()
        self.optimiser = dependancy.create_optimiser(self.model)

        # Useful for FL experiments
        self.num_samples = len(train_loader.dataset)

    def train_model(self, epochs=2):

        self.model.train()

        epoch_losses = []

        for epoch in range(epochs):

            run_loss = 0.0

            for images, labels in self.train_loader:

                outputs = self.model(images)

                loss = self.criterion(
                    outputs,
                    labels
                )

                self.optimiser.zero_grad()

                loss.backward()

                self.optimiser.step()

                run_loss += loss.item()

            average_loss = run_loss / len(self.train_loader)

            epoch_losses.append(average_loss)

        return epoch_losses[-1]

    def get_weights(self):

        return deepcopy(self.model.state_dict())

    def set_weights(self, weights):

        self.model.load_state_dict(weights)