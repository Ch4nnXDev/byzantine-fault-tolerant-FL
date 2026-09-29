
from clients.client import Client
from server.server import Server
from datasets.dataLoader import DataManager
from metrics.metrics import Metrics

from visualization.plotter import ExperimentPlotter
from reporting.report_generator import ResearchReportGenerator

from torchvision import datasets

from config.config_loader import load_config
from utils.seed import set_seed


class Main:

    def __init__(self, config):

        self.config = config

        seed = config["experiment"]["seed"]

        self.server = Server()

        self.clients = []

        self.data_manager = DataManager(
            datasets.MNIST,
            seed=seed
        )

        self.metrics = Metrics(
            experiment_name=config["experiment"]["name"]
        )

    # =========================================================
    # Create clients
    # =========================================================

    def create_clients(self, num_clients):

        dataset_config = self.config["dataset"]

        batch_size = dataset_config["batch_size"]

        partition_type = (
            dataset_config["partition"]["type"]
        )

        client_loaders = (
            self.data_manager.create_client_loaders(
                num_clients=num_clients,
                batch_size=batch_size,
                partition_type=partition_type
            )
        )

        learning_rate = (
            self.config["federated_learning"]
            ["learning_rate"]
        )

        for train_loader in client_loaders:

            client = Client(
                train_loader,
                learning_rate=learning_rate
            )

            self.clients.append(client)

    # =========================================================
    # Run experiment
    # =========================================================

    def run(self):

        fl_config = (
            self.config["federated_learning"]
        )

        dataset_config = (
            self.config["dataset"]
        )

        byzantine_config = (
            self.config["byzantine"]
        )

        topology_config = (
            self.config["topology"]
        )

        defense_config = (
            self.config["defense"]
        )

        rounds = fl_config["rounds"]

        local_epochs = fl_config["local_epochs"]

        # =====================================================
        # Record experiment configuration
        # =====================================================

        self.metrics.set_metadata(

            num_clients=len(self.clients),

            num_rounds=rounds,

            local_epochs=local_epochs,

            batch_size=dataset_config["batch_size"],

            learning_rate=fl_config["learning_rate"],

            byzantine_clients=(
                byzantine_config["num_nodes"]
            ),

            attack=(
                byzantine_config["attack"]
            ),

            aggregation_method="FedAvg",

            dataset=dataset_config["name"],

            model="CNN",

            seed=self.config["experiment"]["seed"],

            partition_type=(
                dataset_config["partition"]["type"]
            ),

            topology=(
                topology_config["type"]
            ),

            defense_enabled=(
                defense_config["enabled"]
            ),

            defense_method=(
                defense_config["method"]
            )
        )

        # =====================================================
        # Start total experiment timer
        # =====================================================

        self.metrics.start_experiment_timer()

        # =====================================================
        # Federated learning rounds
        # =====================================================

        for round_number in range(rounds):

            current_round = round_number + 1

            print(
                f"\n========== Round "
                f"{current_round} =========="
            )

            # ---------------------------------------------
            # Start round timer
            # ---------------------------------------------

            self.metrics.start_round_timer()

            # ---------------------------------------------
            # Send global model to clients
            # ---------------------------------------------

            self.server.send_model(
                self.clients
            )

            # ---------------------------------------------
            # Local client training
            # ---------------------------------------------

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

            # ---------------------------------------------
            # Receive client models
            # ---------------------------------------------

            for client in self.clients:

                self.server.receive_weights(
                    client
                )

            # ---------------------------------------------
            # Aggregate client models
            # ---------------------------------------------

            self.server.aggregate()

            # ---------------------------------------------
            # Evaluate global model
            # ---------------------------------------------

            accuracy, loss = (
                self.server.evaluate()
            )

            self.metrics.record_global(
                current_round,
                accuracy,
                loss
            )

            # ---------------------------------------------
            # Finish round timer
            # ---------------------------------------------

            round_duration = (
                self.metrics.finish_round_timer()
            )

            # ---------------------------------------------
            # Print round results
            # ---------------------------------------------

            print(
                f"Global Accuracy: "
                f"{accuracy:.2%}"
            )

            print(
                f"Global Loss: "
                f"{loss:.4f}"
            )

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

            print(
                f"Round Duration: "
                f"{round_duration:.2f} seconds"
            )

        # =====================================================
        # Finish experiment timer
        # =====================================================

        total_duration = (
            self.metrics.finish_experiment_timer()
        )

        print(
            f"\nTotal Experiment Duration: "
            f"{total_duration:.2f} seconds"
        )

        # =====================================================
        # Print experiment analytics
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

        plot_files = plotter.generate_all()

        for name, path in plot_files.items():

            if path:

                print(
                    f"Generated {name}: {path}"
                )

        # =====================================================
        # Generate research report
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
            "\n" + "=" * 60
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
    # Print metrics
    # =========================================================

    def print_metrics(self):

        print(
            "\n" + "=" * 60
        )

        print(
            "FINAL EXPERIMENT ANALYTICS"
        )

        print(
            "=" * 60
        )

        # =====================================================
        # Global metrics
        # =====================================================

        print(
            "\nGLOBAL METRICS"
        )

        global_metrics = (
            self.metrics.get_global_metrics()
        )

        for i in range(
            len(global_metrics["rounds"])
        ):

            round_duration = 0.0

            if i < len(
                global_metrics["round_durations"]
            ):

                round_duration = (
                    global_metrics[
                        "round_durations"
                    ][i]
                )

            print(

                f"Round "
                f"{global_metrics['rounds'][i]} | "

                f"Accuracy: "
                f"{global_metrics['accuracy'][i]:.2%} | "

                f"Accuracy Change: "
                f"{global_metrics['accuracy_change'][i]:+.2%} | "

                f"Loss: "
                f"{global_metrics['loss'][i]:.4f} | "

                f"Client Mean Loss: "
                f"{global_metrics['client_mean_loss'][i]:.4f} | "

                f"Client Std: "
                f"{global_metrics['client_loss_std'][i]:.4f} | "

                f"Duration: "
                f"{round_duration:.2f}s"
            )

        # =====================================================
        # Convergence analysis
        # =====================================================

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
                f"Best Accuracy: "
                f"{convergence['best_accuracy']:.2%}"
            )

            print(
                f"Best Accuracy Round: "
                f"{convergence['best_accuracy_round']}"
            )

            print(
                f"Accuracy Gain: "
                f"{convergence['accuracy_gain']:+.2%}"
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
                f"Average Accuracy Change: "
                f"{convergence['average_accuracy_change']:.4f}"
            )

            print(
                f"Accuracy Volatility: "
                f"{convergence['accuracy_volatility']:.4f}"
            )

            print(
                f"Average Loss Change: "
                f"{convergence['average_loss_change']:.4f}"
            )

            print(
                f"Loss Volatility: "
                f"{convergence['loss_volatility']:.4f}"
            )

            print(
                f"Average Round Duration: "
                f"{convergence['average_round_duration']:.2f}s"
            )

            print(
                f"Minimum Round Duration: "
                f"{convergence['min_round_duration']:.2f}s"
            )

            print(
                f"Maximum Round Duration: "
                f"{convergence['max_round_duration']:.2f}s"
            )

            print(
                f"Total Experiment Duration: "
                f"{convergence['total_experiment_duration']:.2f}s"
            )

        # =====================================================
        # Client training losses
        # =====================================================

        print(
            "\nCLIENT TRAINING LOSSES"
        )

        for (
            client_id,
            losses
        ) in self.metrics.get_client_metrics().items():

            print(
                f"\nClient {client_id}:"
            )

            for (
                round_number,
                loss
            ) in enumerate(
                losses,
                start=1
            ):

                print(
                    f"  Round "
                    f"{round_number}: "
                    f"{loss:.4f}"
                )

        # =====================================================
        # Client convergence
        # =====================================================

        print(
            "\nCLIENT CONVERGENCE"
        )

        client_convergence = (
            self.metrics
            .get_client_convergence_metrics()
        )

        for (
            client_id,
            data
        ) in client_convergence.items():

            print(

                f"Client {client_id} | "

                f"Initial Loss: "
                f"{data['initial_loss']:.4f} | "

                f"Final Loss: "
                f"{data['final_loss']:.4f} | "

                f"Loss Reduction: "
                f"{data['loss_reduction']:.4f}"
            )


# =============================================================
# Entry point
# =============================================================

if __name__ == "__main__":

    config = load_config(
        "config/baseline.yaml"
    )

    seed = (
        config["experiment"]["seed"]
    )

    set_seed(seed)

    app = Main(
        config
    )

    num_clients = (
        config["federated_learning"]
        ["num_clients"]
    )

    app.create_clients(
        num_clients
    )

    app.run()