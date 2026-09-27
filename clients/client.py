
import uuid
from copy import deepcopy

from dependancyService.dependancyService import DependancyService


class Client:

    def __init__(
        self,
        train_loader,
        learning_rate=0.01
    ):

        dependancy = DependancyService()

        # =====================================================
        # Client identity
        # =====================================================

        self.id = uuid.uuid4()

        # =====================================================
        # Client dataset
        # =====================================================

        self.train_loader = train_loader

        # =====================================================
        # Training components
        # =====================================================

        self.criterion = (
            dependancy.create_loss_function()
        )

        self.model = (
            dependancy.create_model()
        )

        self.optimiser = (
            dependancy.create_optimiser(
                self.model,
                learning_rate=learning_rate
            )
        )

        # =====================================================
        # Experiment configuration
        # =====================================================

        self.learning_rate = learning_rate

        # Number of samples belonging to this client
        self.num_samples = (
            len(train_loader.dataset)
        )

    # =========================================================
    # Local training
    # =========================================================

    def train_model(self, epochs=2):

        self.model.train()

        epoch_losses = []

        for epoch in range(epochs):

            run_loss = 0.0

            for images, labels in self.train_loader:

                # -------------------------------------------------
                # Forward pass
                # -------------------------------------------------

                outputs = self.model(images)

                # -------------------------------------------------
                # Calculate loss
                # -------------------------------------------------

                loss = self.criterion(
                    outputs,
                    labels
                )

                # -------------------------------------------------
                # Clear previous gradients
                # -------------------------------------------------

                self.optimiser.zero_grad()

                # -------------------------------------------------
                # Backpropagation
                # -------------------------------------------------

                loss.backward()

                # -------------------------------------------------
                # Update model
                # -------------------------------------------------

                self.optimiser.step()

                run_loss += loss.item()

            # -------------------------------------------------
            # Average epoch loss
            # -------------------------------------------------

            average_loss = (
                run_loss /
                len(self.train_loader)
            )

            epoch_losses.append(
                average_loss
            )

        return epoch_losses[-1]

    # =========================================================
    # Model weights
    # =========================================================

    def get_weights(self):

        return deepcopy(
            self.model.state_dict()
        )

    # =========================================================
    # Receive global model
    # =========================================================

    def set_weights(self, weights):

        self.model.load_state_dict(
            weights
        )

