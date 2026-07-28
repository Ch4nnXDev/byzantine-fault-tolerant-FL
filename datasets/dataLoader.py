from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split


class DataManager:
    def __init__(self, dataset):
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

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size
        )

        return train_loader, test_loader

    def create_client_loaders(self, num_clients, batch_size=32):

        transform = transforms.ToTensor()

        train_dataset = self.dataset(
            root="data",
            train=True,
            download=True,
            transform=transform
        )

        # Number of samples each client receives
        partition_size = len(train_dataset) // num_clients

        # Create a list of partition sizes
        lengths = [partition_size] * num_clients

        # If the dataset size isn't perfectly divisible,
        # give the remaining samples to the last client
        lengths[-1] += len(train_dataset) - sum(lengths)

        # Split the dataset
        client_datasets = random_split(
            train_dataset,
            lengths
        )

        client_loaders = []

        for dataset in client_datasets:

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True
            )

            client_loaders.append(loader)

        return client_loaders