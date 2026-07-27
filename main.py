from clients.client import Client
from server.server import Server

class Main:
    def __init__(self):
        self.server = Server()
        self.clients = []
        
        
    def create_clients(self, num):
        for i in range(num):
            client = Client()
            self.clients.append(client)
        
            
    def run(self, rounds):
        loss_collect = []
        for round in range(rounds):
            
            print(f"Epoch {round + 1}")
            
  
            
            self.server.send_model(self.clients)
            
            for client in self.clients:
                gg = client.train_model()
                print(gg)
                
                
                
            for clients in self.clients:
                
                self.server.receive_weights(
                    clients.get_weights()
                )
                
            self.server.aggregate()
            
            accuracy, loss = self.server.evaluate()


            print(
                f"Global Accuracy: {accuracy:.2%}"
            )

            print(
                f"Global Loss: {loss:.4f}"
            )
            
    

if __name__ == "__main__":

    app = Main()

    app.create_clients(5)

    app.run(5)
    
                
            
        
        
   
        
        
        
 