from clients.client import Client
from clients.attacker import Attacker
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

        # =====================================================
        # Experiment configuration
        # =====================================================

        experiment_config = (
            config["experiment"]
        )

        fl_config = (
            config["federated_learning"]
        )

        dataset_config = (
            config["dataset"]
        )

        byzantine_config = (
            config["byzantine"]
        )

        topology_config = (
            config["topology"]
        )

        defense_config = (
            config["defense"]
        )

        seed = experiment_config["seed"]

        # =====================================================
        # Reproducibility
        # =====================================================

        set_seed(seed)

        # =====================================================
        # Server
        # =====================================================

        self.server = Server()

        # =====================================================
        # Clients
        # =====================================================

        self.clients = []

        # =====================================================
        # Dataset
        # =====================================================

        self.data_manager = DataManager(
            datasets.MNIST,
            seed=seed
        )

        # =====================================================
        # Metrics
        # =====================================================

        self.metrics = Metrics(
            experiment_name=(
                experiment_config["name"]
            )
        )

        # =====================================================
        # Store configuration values
        #
        # These make the experiment metadata explicit.
        # =====================================================

        self.num_clients = (
            fl_config["num_clients"]
        )

        self.rounds = (
            fl_config["rounds"]
        )

        self.local_epochs = (
            fl_config["local_epochs"]
        )

        self.learning_rate = (
            fl_config["learning_rate"]
        )

        self.dataset_name = (
            dataset_config["name"]
        )

        self.batch_size = (
            dataset_config["batch_size"]
        )

        self.partition_type = (
            dataset_config["partition"]["type"]
        )

        self.byzantine_enabled = (
            byzantine_config["enabled"]
        )

        self.num_attackers = (
            byzantine_config["num_nodes"]
        )

        self.attack_type = (
            byzantine_config["attack"]
        )
        
        self.attack_strength = (
            byzantine_config["attack_strength"]
        )

        self.topology_type = (
            topology_config["type"]
        )

        self.defense_enabled = (
            defense_config["enabled"]
        )

        self.defense_method = (
            defense_config["method"]
        )

    # =========================================================
    # CLIENT CREATION
    # =========================================================

    def create_clients(self, num_clients):

        # -----------------------------------------------------
        # Create client dataset partitions
        # -----------------------------------------------------

        client_loaders = (
            self.data_manager.create_client_loaders(
                num_clients=num_clients,
                batch_size=self.batch_size,
                partition_type=self.partition_type
            )
        )

        # -----------------------------------------------------
        # Byzantine configuration
        # -----------------------------------------------------

        num_attackers = (
            self.num_attackers
        )

        if not self.byzantine_enabled:

            num_attackers = 0

        # -----------------------------------------------------
        # Validate configuration
        # -----------------------------------------------------

        if num_attackers < 0:

            raise ValueError(
                "Number of Byzantine nodes "
                "cannot be negative."
            )

        if num_attackers > num_clients:

            raise ValueError(
                "Number of Byzantine nodes "
                "cannot exceed the number "
                "of clients."
            )

        # -----------------------------------------------------
        # Create clients
        # -----------------------------------------------------

        for client_index, train_loader in enumerate(
            client_loaders
        ):

            # -------------------------------------------------
            # Deterministic attacker placement
            #
            # For the initial experiments, the first N
            # clients are Byzantine.
            #
            # This will later be replaced by topology-aware
            # attacker placement.
            # -------------------------------------------------

            if client_index < num_attackers:

                client = Attacker(
                    train_loader=train_loader,
                    learning_rate=self.learning_rate,
                    attack_type=self.attack_type,
                    attack_strength=self.attack_strength
                )

            else:

                client = Client(
                    train_loader=train_loader,
                    learning_rate=self.learning_rate
                )

            self.clients.append(client)

        # -----------------------------------------------------
        # Print client configuration
        # -----------------------------------------------------

        print("\nClient configuration:")

        for client_index, client in enumerate(
            self.clients
        ):

            if isinstance(client, Attacker):

                print(
                    f"Client {client_index + 1}: "
                    f"BYZANTINE "
                    f"(attack={client.attack_type})"
                )

            else:

                print(
                    f"Client {client_index + 1}: "
                    f"HONEST"
                )

    # =========================================================
    # METADATA
    # =========================================================

    def configure_metrics_metadata(self):

        # -----------------------------------------------------
        # Determine aggregation method
        #
        # Current platform uses FedAvg through Server.
        # -----------------------------------------------------

        aggregation_method = "FedAvg"

        # -----------------------------------------------------
        # Determine model name
        #
        # DependancyService creates the model, so we obtain
        # the class name from one client after clients exist.
        # -----------------------------------------------------

        model_name = None

        if self.clients:

            model_name = (
                self.clients[0]
                .model
                .__class__
                .__name__
            )

        # -----------------------------------------------------
        # Configure experiment metadata
        # -----------------------------------------------------

        self.metrics.set_metadata(

            num_clients=self.num_clients,

            num_rounds=self.rounds,

            local_epochs=self.local_epochs,

            byzantine_clients=(
                self.num_attackers
                if self.byzantine_enabled
                else 0
            ),

            aggregation_method=(
                aggregation_method
            ),

            dataset=self.dataset_name,

            model=model_name,

            seed=self.config[
                "experiment"
            ]["seed"],

            batch_size=self.batch_size,

            learning_rate=self.learning_rate,

            partition_type=self.partition_type,

            topology=self.topology_type,

            defense_enabled=self.defense_enabled,

            defense_method=self.defense_method,

            attack=(
                self.attack_type
                if self.byzantine_enabled
                else "none"
            ),
            
            attack_strength=(
                self.attack_strength
                if self.byzantine_enabled
                else None
            )
        )

    # =========================================================
    # RUN EXPERIMENT
    # =========================================================

    def run(self):

        print("\n========================================")
        print("Starting Experiment")
        print("========================================")

        print(
            "Experiment:",
            self.config["experiment"]["name"]
        )

        print(
            "Seed:",
            self.config["experiment"]["seed"]
        )

        print(
            "Clients:",
            self.num_clients
        )

        print(
            "Rounds:",
            self.rounds
        )

        print(
            "Local epochs:",
            self.local_epochs
        )

        print(
            "Learning rate:",
            self.learning_rate
        )

        print(
            "Dataset:",
            self.dataset_name
        )

        print(
            "Partition:",
            self.partition_type
        )

        print(
            "Byzantine enabled:",
            self.byzantine_enabled
        )

        print(
            "Byzantine nodes:",
            (
                self.num_attackers
                if self.byzantine_enabled
                else 0
            )
        )

        print(
            "Attack:",
            (
                self.attack_type
                if self.byzantine_enabled
                else "none"
            )
        )
        
        print(
            "Attack Strength:",
            (
                self.attack_strength
                if self.byzantine_enabled
                else "none"
            )
        )

        print(
            "Topology:",
            self.topology_type
        )

        print(
            "Defense:",
            self.defense_enabled
        )

        print(
            "Defense method:",
            self.defense_method
        )

        # =====================================================
        # Create clients
        # =====================================================

        self.create_clients(
            num_clients=self.num_clients
        )

        # =====================================================
        # Configure metrics metadata
        # =====================================================

        self.configure_metrics_metadata()

        # =====================================================
        # Start experiment timer
        # =====================================================

        self.metrics.start_experiment_timer()

        # =====================================================
        # Federated learning rounds
        # =====================================================

        for round_number in range(
            1,
            self.rounds + 1
        ):

            print(
                f"\n========== Round "
                f"{round_number}/{self.rounds} =========="
            )

            # -------------------------------------------------
            # Start round timer
            # -------------------------------------------------

            self.metrics.start_round_timer()

            # -------------------------------------------------
            # Send global model to clients
            # -------------------------------------------------

            self.server.send_model(
                self.clients
            )

            # -------------------------------------------------
            # Local training
            # -------------------------------------------------

            for client in self.clients:

                loss = client.train_model(
                    epochs=self.local_epochs
                )

                self.metrics.record_client(
                    client.id,
                    loss
                )

            # -------------------------------------------------
            # Receive client models
            #
            # For an attacker, get_weights() returns the
            # malicious model generated from its attacked
            # update.
            #
            # The server does not need to know whether the
            # client is honest or Byzantine.
            # -------------------------------------------------

            for client in self.clients:

                self.server.receive_weights(
                    client
                )

            # -------------------------------------------------
            # Aggregate client models
            # -------------------------------------------------

            self.server.aggregate()

            # -------------------------------------------------
            # Evaluate global model
            # -------------------------------------------------

            accuracy, loss = (
                self.server.evaluate()
            )

            # -------------------------------------------------
            # Record global metrics
            # -------------------------------------------------

            self.metrics.record_global(
                round_number,
                accuracy,
                loss
            )

            # -------------------------------------------------
            # Finish round timer
            # -------------------------------------------------

            round_duration = (
                self.metrics.finish_round_timer()
            )

            # -------------------------------------------------
            # Print round results
            # -------------------------------------------------

            print(
                f"Global Accuracy: "
                f"{accuracy:.4f}"
            )

            print(
                f"Global Loss: "
                f"{loss:.4f}"
            )

            print(
                f"Round Duration: "
                f"{round_duration:.2f}s"
            )

        # =====================================================
        # Finish experiment timer
        # =====================================================

        experiment_duration = (
            self.metrics.finish_experiment_timer()
        )

        # =====================================================
        # Save metrics
        # =====================================================

        print("\n========================================")
        print("Saving Experiment Metrics")
        print("========================================")

        metric_files = (
            self.metrics.save_all()
        )

        print(
            "Experiment directory:",
            metric_files[
                "experiment_directory"
            ]
        )

        print(
            "Metrics JSON:",
            metric_files["json"]
        )

        print(
            "Global CSV:",
            metric_files["global_csv"]
        )

        print(
            "Client CSV:",
            metric_files["client_csv"]
        )

        print(
            "Timing CSV:",
            metric_files["timing_csv"]
        )

        # =====================================================
        # Generate plots
        # =====================================================

        print("\n========================================")
        print("Generating Plots")
        print("========================================")

        plotter = ExperimentPlotter(
            self.metrics
        )

        plot_files = plotter.generate_all()

        # =====================================================
        # Generate research report
        # =====================================================

        print("\n========================================")
        print("Generating Research Report")
        print("========================================")

        report_generator = (
            ResearchReportGenerator(
                self.metrics,
                plot_files
            )
        )

        report_generator.generate()

        # =====================================================
        # Final experiment summary
        # =====================================================

        convergence = (
            self.metrics.get_convergence_metrics()
        )

        print("\n========================================")
        print("Experiment Completed")
        print("========================================")

        print(
            f"Final Accuracy: "
            f"{convergence['final_accuracy']:.4f}"
        )

        print(
            f"Best Accuracy: "
            f"{convergence['best_accuracy']:.4f}"
        )

        print(
            f"Best Accuracy Round: "
            f"{convergence['best_accuracy_round']}"
        )

        print(
            f"Final Loss: "
            f"{convergence['final_loss']:.4f}"
        )

        print(
            f"Accuracy Gain: "
            f"{convergence['accuracy_gain']:.4f}"
        )

        print(
            f"Loss Reduction: "
            f"{convergence['loss_reduction']:.4f}"
        )

        print(
            f"Experiment Duration: "
            f"{experiment_duration:.2f}s"
        )

        print(
            "\nResults saved to:"
        )

        print(
            self.metrics.output_dir
        )

    # =========================================================
    # PRINT METRICS
    # =========================================================

    def print_metrics(self):

        print("\n========================================")
        print("Experiment Metrics")
        print("========================================")

        global_metrics = (
            self.metrics.get_global_metrics()
        )

        convergence_metrics = (
            self.metrics.get_convergence_metrics()
        )

        print(
            "\nGlobal Accuracy:"
        )

        for round_number, accuracy in zip(
            global_metrics["rounds"],
            global_metrics["accuracy"]
        ):

            print(
                f"Round {round_number}: "
                f"{accuracy:.4f}"
            )

        print(
            "\nGlobal Loss:"
        )

        for round_number, loss in zip(
            global_metrics["rounds"],
            global_metrics["loss"]
        ):

            print(
                f"Round {round_number}: "
                f"{loss:.4f}"
            )

        print(
            "\nConvergence:"
        )

        for key, value in (
            convergence_metrics.items()
        ):

            print(
                f"{key}: {value}"
            )


# =============================================================
# ENTRY POINT
# =============================================================

def main():

    # ---------------------------------------------------------
    # Select experiment configuration
    #
    # Keep baseline.yaml for EXP01.
    #
    # Change this path to:
    #
    # config/exp02_byzantine_sign_flip.yaml
    #
    # when running EXP02.
    # ---------------------------------------------------------

    config = load_config(
        "config/exp03.1_byzantine_sign_flip_multiple.yaml"
    )

    experiment = Main(
        config
    )

    experiment.run()


if __name__ == "__main__":

    main()