from torchvision import transforms

from torch.utils.data import DataLoader
from torch.utils.data import random_split

import torch


class DataManager:

    def __init__(
        self,
        dataset,
        seed=42
    ):

        self.dataset = dataset
        self.seed = seed

    # =========================================================
    # Dataset loading
    # =========================================================

    def load_dataset(
        self,
        batch_size=32
    ):

        transform = transforms.ToTensor()

        # -----------------------------------------------------
        # Training dataset
        # -----------------------------------------------------

        train_dataset = self.dataset(
            root="data",
            train=True,
            download=True,
            transform=transform
        )

        # -----------------------------------------------------
        # Test dataset
        # -----------------------------------------------------

        test_dataset = self.dataset(
            root="data",
            train=False,
            download=True,
            transform=transform
        )

        # -----------------------------------------------------
        # DataLoaders
        # -----------------------------------------------------

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        return train_loader, test_loader

    # =========================================================
    # Client dataset partitioning
    # =========================================================

    def create_client_loaders(
        self,
        num_clients,
        batch_size=32,
        partition_type="iid"
    ):

        # -----------------------------------------------------
        # Currently supported partition
        # -----------------------------------------------------

        if partition_type != "iid":

            raise ValueError(
                f"Unsupported partition type: "
                f"{partition_type}"
            )

        transform = transforms.ToTensor()

        # -----------------------------------------------------
        # Load training dataset
        # -----------------------------------------------------

        train_dataset = self.dataset(
            root="data",
            train=True,
            download=True,
            transform=transform
        )

        # -----------------------------------------------------
        # Calculate partition sizes
        # -----------------------------------------------------

        partition_size = (
            len(train_dataset) //
            num_clients
        )

        lengths = (
            [partition_size] *
            num_clients
        )

        # Give remaining samples to last client
        lengths[-1] += (
            len(train_dataset) -
            sum(lengths)
        )

        # -----------------------------------------------------
        # Create deterministic generator
        # -----------------------------------------------------

        generator = torch.Generator()

        generator.manual_seed(
            self.seed
        )

        # -----------------------------------------------------
        # Partition dataset
        # -----------------------------------------------------

        client_datasets = random_split(
            train_dataset,
            lengths,
            generator=generator
        )

        # -----------------------------------------------------
        # Create client DataLoaders
        # -----------------------------------------------------

        client_loaders = []

        for client_dataset in client_datasets:

            loader = DataLoader(
                client_dataset,
                batch_size=batch_size,
                shuffle=True
            )

            client_loaders.append(
                loader
            )

        return client_loaders