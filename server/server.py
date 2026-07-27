from dependancyService.dependancyService import DependancyService
import torch

class Server:
    

    def __init__(self):
        
        dependancy = DependancyService()
        self.model = dependancy.create_model()
        self.train_loader, self.test_loader = dependancy.create_loader()
        self.criterion = dependancy.create_loss_function()
        
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
                    for weights in self.received_weights
                    ]
                
            ).mean(dim=0)
        self.model.load_state_dict(
            new_weights
        )


        self.client_weights = []
    
    def evaluate(self):
        self.model.eval()
        correct = 0
        loss = 0
        total = 0
        total_loss = 0
        
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

        average_loss = total_loss / len(self.test_loader)


        return accuracy, average_loss
        