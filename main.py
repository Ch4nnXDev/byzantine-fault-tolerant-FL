from clients.client import Client

class Main:
    def __init__(self):
        pass
    
    def create_clients(self, num):
        clients = []
        for i in range(num):
           client = Client()
           client.add(client)
        return clients
    
    
            
        
        
        
 