from copy import deepcopy

from dependancyService.dependancyService import DependancyService
import torch


class Server:

    def __init__(self):

        dependancy = DependancyService()

        self.model = dependancy.create_model()

        self.train_loader, self.test_loader = (
            dependancy.create_loader()
        )

        self.criterion = dependancy.create_loss_function()

        self.received_updates = []

    def get_model_weights(self):

        return deepcopy(
            self.model.state_dict()
        )

    def send_model(self, clients):

        weights = self.get_model_weights()

        for client in clients:
            client.set_weights(weights)

    def receive_weights(self, client):

        self.received_updates.append({
            "client_id": client.id,
            "weights": client.get_weights(),
            "num_samples": client.num_samples
        })

    def aggregate(self):

        total_samples = sum(
            update["num_samples"]
            for update in self.received_updates
        )

        new_weights = {}

        for key in self.model.state_dict().keys():

            weighted_sum = None

            for update in self.received_updates:

                weight = (
                    update["num_samples"]
                    / total_samples
                )

                client_weight = (
                    update["weights"][key].float()
                )

                contribution = (
                    client_weight * weight
                )

                if weighted_sum is None:
                    weighted_sum = contribution
                else:
                    weighted_sum += contribution

            new_weights[key] = weighted_sum

        self.model.load_state_dict(new_weights)

        self.received_updates = []

    def evaluate(self):

        self.model.eval()

        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():

            for images, labels in self.test_loader:

                outputs = self.model(images)

                loss = self.criterion(
                    outputs,
                    labels
                )

                total_loss += loss.item()

                predicted = torch.argmax(
                    outputs,
                    dim=1
                )

                total += labels.size(0)

                correct += (
                    predicted == labels
                ).sum().item()

        accuracy = correct / total

        average_loss = (
            total_loss / len(self.test_loader)
        )

        return accuracy, average_loss