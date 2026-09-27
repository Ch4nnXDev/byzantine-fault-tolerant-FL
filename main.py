
from clients.client import Client
from server.server import Server
from datasets.dataLoader import DataManager
from metrics.metrics import Metrics

from visualization.plotter import ExperimentPlotter
from reporting.report_generator import ResearchReportGenerator

from torchvision import datasets

from config.config_loader import load_config


class Main:

    def __init__(self, config):

        self.config = config

        self.server = Server()

        self.clients = []

        self.data_manager = DataManager(
            datasets.MNIST
        )

        self.metrics = Metrics(
            experiment_name=config["experiment"]["name"]
        )

    def create_clients(self, num):

        client_loaders = (
            self.data_manager.create_client_loaders(
                num
            )
        )

        # Get learning rate from experiment configuration
        learning_rate = self.config[
            "federated_learning"
        ]["learning_rate"]

        for train_loader in client_loaders:

            client = Client(
                train_loader,
                learning_rate=learning_rate
            )

            self.clients.append(client)

    def run(self):

        # =====================================================
        # Experiment configuration
        # =====================================================

        fl_config = self.config[
            "federated_learning"
        ]

        rounds = fl_config["rounds"]

        local_epochs = fl_config[
            "local_epochs"
        ]

        self.metrics.set_metadata(

            num_clients=len(self.clients),

            num_rounds=rounds,

            local_epochs=local_epochs,

            byzantine_clients=self.config[
                "byzantine"
            ]["num_nodes"],

            aggregation_method="FedAvg",

            dataset=self.config[
                "dataset"
            ]["name"],

            model="Current Model",

            seed=self.config[
                "experiment"
            ]["seed"]

        )

        # =====================================================
        # Federated training
        # =====================================================

        for round_number in range(rounds):

            current_round = (
                round_number + 1
            )

            print(
                f"\n========== Round "
                f"{current_round} =========="
            )

            # -------------------------------------------------
            # 1. Send global model
            # -------------------------------------------------

            self.server.send_model(
                self.clients
            )

            # -------------------------------------------------
            # 2. Local training
            # -------------------------------------------------

            for client in self.clients:

                loss = client.train_model(
                    epochs=local_epochs
                )

                self.metrics.record_client(
                    client.id,
                    loss
                )

                print(
                    f"Client {client.id} "
                    f"Loss: {loss:.4f}"
                )

            # -------------------------------------------------
            # 3. Receive updates
            # -------------------------------------------------

            for client in self.clients:

                self.server.receive_weights(
                    client
                )

            # -------------------------------------------------
            # 4. Aggregate
            # -------------------------------------------------

            self.server.aggregate()

            # -------------------------------------------------
            # 5. Evaluate
            # -------------------------------------------------

            accuracy, loss = (
                self.server.evaluate()
            )

            self.metrics.record_global(

                current_round,

                accuracy,

                loss

            )

            print(
                f"Global Accuracy: "
                f"{accuracy:.2%}"
            )

            print(
                f"Global Loss: "
                f"{loss:.4f}"
            )

            # -------------------------------------------------
            # Round analytics
            # -------------------------------------------------

            print(
                f"Client Mean Loss: "
                f"{self.metrics.client_mean_loss[-1]:.4f}"
            )

            print(
                f"Client Loss Std: "
                f"{self.metrics.client_loss_std[-1]:.4f}"
            )

            print(
                f"Client Loss Range: "
                f"{self.metrics.client_loss_range[-1]:.4f}"
            )

        # =====================================================
        # Final analytics
        # =====================================================

        self.print_metrics()

        # =====================================================
        # Save experiment metrics
        # =====================================================

        print(
            "\nSaving experiment metrics..."
        )

        files = self.metrics.save_all()

        print(
            "\nMetrics saved:"
        )

        print(
            files["experiment_directory"]
        )

        # =====================================================
        # Generate visualisations
        # =====================================================

        print(
            "\nGenerating experiment visualisations..."
        )

        plotter = ExperimentPlotter(
            self.metrics
        )

        plot_files = (
            plotter.generate_all()
        )

        for name, path in plot_files.items():

            if path:

                print(
                    f"Generated {name}: {path}"
                )

        # =====================================================
        # Generate research document
        # =====================================================

        print(
            "\nGenerating research report..."
        )

        report_generator = (
            ResearchReportGenerator(
                self.metrics,
                plot_files
            )
        )

        report_path = (
            report_generator.generate()
        )

        print(
            f"Research report: {report_path}"
        )

        # =====================================================
        # Experiment complete
        # =====================================================

        print(
            "\n"
            + "=" * 60
        )

        print(
            "EXPERIMENT COMPLETE"
        )

        print(
            "=" * 60
        )

        print(
            "Results directory:"
        )

        print(
            self.metrics.output_dir
        )

    # =========================================================
    # Print final metrics
    # =========================================================

    def print_metrics(self):

        print(
            "\n"
            + "=" * 60
        )

        print(
            "FINAL EXPERIMENT ANALYTICS"
        )

        print(
            "=" * 60
        )

        # -----------------------------------------------------
        # Global metrics
        # -----------------------------------------------------

        print(
            "\nGLOBAL METRICS"
        )

        global_metrics = (
            self.metrics.get_global_metrics()
        )

        for i in range(
            len(global_metrics["rounds"])
        ):

            print(

                f"Round "
                f"{global_metrics['rounds'][i]} | "

                f"Accuracy: "
                f"{global_metrics['accuracy'][i]:.2%} | "

                f"Loss: "
                f"{global_metrics['loss'][i]:.4f} | "

                f"Mean Client Loss: "
                f"{global_metrics['client_mean_loss'][i]:.4f} | "

                f"Client Std: "
                f"{global_metrics['client_loss_std'][i]:.4f}"

            )

        # -----------------------------------------------------
        # Convergence
        # -----------------------------------------------------

        convergence = (
            self.metrics.get_convergence_metrics()
        )

        print(
            "\nCONVERGENCE ANALYSIS"
        )

        if convergence:

            print(
                f"Initial Accuracy: "
                f"{convergence['initial_accuracy']:.2%}"
            )

            print(
                f"Final Accuracy: "
                f"{convergence['final_accuracy']:.2%}"
            )

            print(
                f"Accuracy Gain: "
                f"{convergence['accuracy_gain']:.2%}"
            )

            print(
                f"Initial Loss: "
                f"{convergence['initial_loss']:.4f}"
            )

            print(
                f"Final Loss: "
                f"{convergence['final_loss']:.4f}"
            )

            print(
                f"Loss Reduction: "
                f"{convergence['loss_reduction']:.4f}"
            )

            print(
                f"Loss Reduction %: "
                f"{convergence['loss_reduction_percent']:.2f}%"
            )

            print(
                f"Largest Accuracy Improvement: "
                f"{convergence['max_accuracy_improvement']:.2%}"
            )

        # -----------------------------------------------------
        # Client metrics
        # -----------------------------------------------------

        print(
            "\nCLIENT TRAINING LOSSES"
        )

        for client_id, losses in (
            self.metrics.get_client_metrics().items()
        ):

            print(
                f"\nClient {client_id}:"
            )

            for round_number, loss in enumerate(
                losses,
                start=1
            ):

                print(
                    f"  Round {round_number}: "
                    f"{loss:.4f}"
                )


if __name__ == "__main__":

    config = load_config(
        "config/baseline.yaml"
    )

    app = Main(config)

    num_clients = (
        config["federated_learning"][
            "num_clients"
        ]
    )

    app.create_clients(
        num_clients
    )

    app.run()
