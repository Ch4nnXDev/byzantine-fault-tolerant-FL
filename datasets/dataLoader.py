from torchvision import transforms
from torch.utils.data import DataLoader

class DataManager:
    def __init__(self,dataset):
        self.dataset = dataset
        
    def load_dataset(self, batch_size=32):
        
        transform = transforms.ToTensor()
        
        train_dataset = self.dataset(
            root="data",
            train=True,
            download=True,
            transform=transform
            
        )
        
        test_dataset = self.dataset(
            root="data",
            train=False,
            download=True,
            transform=transform
        )
        
        train_Loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
            
        )
        
        test_Loader = DataLoader(
            test_dataset,
            batch_size=batch_size
        )
        
        return train_Loader, test_Loader
    
        
        
        