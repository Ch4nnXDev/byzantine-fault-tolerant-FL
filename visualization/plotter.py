import os
import matplotlib.pyplot as plt


class ExperimentPlotter:

    def __init__(self, metrics):

        self.metrics = metrics
        self.output_dir = metrics.output_dir

    def plot_accuracy(self):

        rounds = self.metrics.rounds
        accuracy = self.metrics.global_accuracy

        if not rounds:
            return None

        path = os.path.join(
            self.output_dir,
            "accuracy.png"
        )

        plt.figure(figsize=(9, 5))

        plt.plot(
            rounds,
            accuracy,
            marker="o"
        )

        plt.title("Global Model Accuracy")
        plt.xlabel("Federated Round")
        plt.ylabel("Accuracy")

        plt.grid(
            True,
            alpha=0.3
        )

        plt.tight_layout()

        plt.savefig(
            path,
            dpi=300
        )

        plt.close()

        return path

    def plot_global_loss(self):

        rounds = self.metrics.rounds
        losses = self.metrics.global_loss

        if not rounds:
            return None

        path = os.path.join(
            self.output_dir,
            "global_loss.png"
        )

        plt.figure(figsize=(9, 5))

        plt.plot(
            rounds,
            losses,
            marker="o"
        )

        plt.title("Global Model Loss")
        plt.xlabel("Federated Round")
        plt.ylabel("Loss")

        plt.grid(
            True,
            alpha=0.3
        )

        plt.tight_layout()

        plt.savefig(
            path,
            dpi=300
        )

        plt.close()

        return path

   
    def plot_client_losses(self):

        if not self.metrics.client_losses:
            return None

        path = os.path.join(
            self.output_dir,
            "client_losses.png"
        )

        plt.figure(figsize=(10, 6))

        for index, (client_id, losses) in enumerate(
            self.metrics.client_losses.items(),
            start=1
        ):

            rounds = range(
                1,
                len(losses) + 1
            )

            plt.plot(
                rounds,
                losses,
                marker="o",
                label=f"Client {index}"
            )

        plt.title(
            "Client Training Loss"
        )

        plt.xlabel(
            "Federated Round"
        )

        plt.ylabel(
            "Training Loss"
        )

        plt.legend()

        plt.grid(
            True,
            alpha=0.3
        )

        plt.tight_layout()

        plt.savefig(
            path,
            dpi=300
        )

        plt.close()

        return path
    
    
    def plot_client_mean_loss(self):

        rounds = self.metrics.rounds
        losses = self.metrics.client_mean_loss

        if not rounds:
            return None

        path = os.path.join(
            self.output_dir,
            "client_mean_loss.png"
        )

        plt.figure(figsize=(9, 5))

        plt.plot(
            rounds,
            losses,
            marker="o"
        )

        plt.title(
            "Mean Client Training Loss"
        )

        plt.xlabel(
            "Federated Round"
        )

        plt.ylabel(
            "Mean Client Loss"
        )

        plt.grid(
            True,
            alpha=0.3
        )

        plt.tight_layout()

        plt.savefig(
            path,
            dpi=300
        )

        plt.close()

        return path
    
    
    def plot_client_loss_variability(self):

        rounds = self.metrics.rounds

        mean_loss = (
            self.metrics.client_mean_loss
        )

        min_loss = (
            self.metrics.client_min_loss
        )

        max_loss = (
            self.metrics.client_max_loss
        )

        if not rounds:
            return None

        path = os.path.join(
            self.output_dir,
            "client_loss_variability.png"
        )

        plt.figure(figsize=(10, 6))

        plt.plot(
            rounds,
            mean_loss,
            marker="o",
            label="Mean"
        )

        plt.plot(
            rounds,
            min_loss,
            marker="o",
            label="Minimum"
        )

        plt.plot(
            rounds,
            max_loss,
            marker="o",
            label="Maximum"
        )

        plt.title(
            "Client Loss Variability"
        )

        plt.xlabel(
            "Federated Round"
        )

        plt.ylabel(
            "Training Loss"
        )

        plt.legend()

        plt.grid(
            True,
            alpha=0.3
        )

        plt.tight_layout()

        plt.savefig(
            path,
            dpi=300
        )

        plt.close()

        return path

    def generate_all(self):

        files = {

            "accuracy":
                self.plot_accuracy(),

            "global_loss":
                self.plot_global_loss(),

            "client_losses":
                self.plot_client_losses(),

            "client_mean_loss":
                self.plot_client_mean_loss(),

            "client_loss_variability":
                self.plot_client_loss_variability()
        }

        return files