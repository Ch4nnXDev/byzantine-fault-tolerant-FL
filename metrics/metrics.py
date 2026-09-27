import csv
import json
import os
import statistics
from datetime import datetime


class Metrics:

    def __init__(
        self,
        experiment_name="experiment",
        output_dir="experiments"
    ):

        # =====================================================
        # Experiment information
        # =====================================================

        self.experiment_name = experiment_name

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        self.experiment_id = (
            f"{experiment_name}_{timestamp}"
        )

        self.output_dir = os.path.join(
            output_dir,
            self.experiment_id
        )

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        # =====================================================
        # Global metrics
        # =====================================================

        self.rounds = []

        self.global_accuracy = []

        self.global_loss = []

        # =====================================================
        # Client metrics
        #
        # {
        #   client_id: [
        #       loss_round_1,
        #       loss_round_2,
        #       ...
        #   ]
        # }
        # =====================================================

        self.client_losses = {}

        # =====================================================
        # Round-level client statistics
        # =====================================================

        self.client_mean_loss = []

        self.client_min_loss = []

        self.client_max_loss = []

        self.client_loss_std = []

        self.client_loss_range = []

        # =====================================================
        # Experiment metadata
        # =====================================================

        self.metadata = {}

    # =========================================================
    # Metadata
    # =========================================================

    def set_metadata(
        self,
        num_clients=None,
        num_rounds=None,
        local_epochs=None,
        byzantine_clients=0,
        aggregation_method=None,
        dataset="MNIST",
        model=None,
        seed=None
    ):

        self.metadata = {

            "experiment_name":
                self.experiment_name,

            "experiment_id":
                self.experiment_id,

            "timestamp":
                datetime.now().isoformat(),

            "dataset":
                dataset,

            "model":
                model,

            "num_clients":
                num_clients,

            "num_rounds":
                num_rounds,

            "local_epochs":
                local_epochs,

            "byzantine_clients":
                byzantine_clients,

            "aggregation_method":
                aggregation_method,

            "random_seed":
                seed
        }

    # =========================================================
    # Record client loss
    # =========================================================

    def record_client(
        self,
        client_id,
        loss
    ):
        
        client_id = str(client_id)

        if client_id not in self.client_losses:

            self.client_losses[client_id] = []

        self.client_losses[client_id].append(
            loss
        )

    # =========================================================
    # Record global metrics
    # =========================================================

    def record_global(
        self,
        round_number,
        accuracy,
        loss
    ):

        self.rounds.append(
            round_number
        )

        self.global_accuracy.append(
            accuracy
        )

        self.global_loss.append(
            loss
        )

        # ---------------------------------------------
        # Calculate client statistics for this round
        # ---------------------------------------------

        current_losses = []

        for losses in self.client_losses.values():

            if len(losses) >= round_number:

                current_losses.append(
                    losses[round_number - 1]
                )

        if current_losses:

            mean_loss = statistics.mean(
                current_losses
            )

            min_loss = min(
                current_losses
            )

            max_loss = max(
                current_losses
            )

            loss_range = (
                max_loss - min_loss
            )

            if len(current_losses) > 1:

                std_loss = statistics.stdev(
                    current_losses
                )

            else:

                std_loss = 0.0

        else:

            mean_loss = 0.0
            min_loss = 0.0
            max_loss = 0.0
            std_loss = 0.0
            loss_range = 0.0

        self.client_mean_loss.append(
            mean_loss
        )

        self.client_min_loss.append(
            min_loss
        )

        self.client_max_loss.append(
            max_loss
        )

        self.client_loss_std.append(
            std_loss
        )

        self.client_loss_range.append(
            loss_range
        )

    # =========================================================
    # Get global metrics
    # =========================================================

    def get_global_metrics(self):

        return {

            "rounds":
                self.rounds,

            "accuracy":
                self.global_accuracy,

            "loss":
                self.global_loss,

            "client_mean_loss":
                self.client_mean_loss,

            "client_min_loss":
                self.client_min_loss,

            "client_max_loss":
                self.client_max_loss,

            "client_loss_std":
                self.client_loss_std,

            "client_loss_range":
                self.client_loss_range
        }

    # =========================================================
    # Get client metrics
    # =========================================================

    def get_client_metrics(self):

        return self.client_losses

    # =========================================================
    # Calculate convergence statistics
    # =========================================================

    def get_convergence_metrics(self):

        if not self.global_accuracy:

            return {}

        initial_accuracy = (
            self.global_accuracy[0]
        )

        final_accuracy = (
            self.global_accuracy[-1]
        )

        initial_loss = (
            self.global_loss[0]
        )

        final_loss = (
            self.global_loss[-1]
        )

        accuracy_gain = (
            final_accuracy -
            initial_accuracy
        )

        loss_reduction = (
            initial_loss -
            final_loss
        )

        if initial_loss != 0:

            loss_reduction_percent = (
                loss_reduction /
                initial_loss
            ) * 100

        else:

            loss_reduction_percent = 0.0

        # ---------------------------------------------
        # Largest accuracy improvement
        # ---------------------------------------------

        accuracy_improvements = []

        for i in range(
            1,
            len(self.global_accuracy)
        ):

            improvement = (
                self.global_accuracy[i]
                -
                self.global_accuracy[i - 1]
            )

            accuracy_improvements.append(
                improvement
            )

        if accuracy_improvements:

            max_accuracy_improvement = max(
                accuracy_improvements
            )

        else:

            max_accuracy_improvement = 0.0

        # ---------------------------------------------
        # Round-to-round stability
        # ---------------------------------------------

        if len(self.global_accuracy) > 1:

            accuracy_changes = [

                abs(
                    self.global_accuracy[i]
                    -
                    self.global_accuracy[i - 1]
                )

                for i in range(
                    1,
                    len(self.global_accuracy)
                )
            ]

            average_accuracy_change = (
                statistics.mean(
                    accuracy_changes
                )
            )

        else:

            average_accuracy_change = 0.0

        return {

            "initial_accuracy":
                initial_accuracy,

            "final_accuracy":
                final_accuracy,

            "accuracy_gain":
                accuracy_gain,

            "initial_loss":
                initial_loss,

            "final_loss":
                final_loss,

            "loss_reduction":
                loss_reduction,

            "loss_reduction_percent":
                loss_reduction_percent,

            "max_accuracy_improvement":
                max_accuracy_improvement,

            "average_accuracy_change":
                average_accuracy_change
        }

    # =========================================================
    # Save JSON
    # =========================================================

    def save_json(self):

        data = {

            "metadata":
                self.metadata,

            "global_metrics":
                self.get_global_metrics(),

            "client_metrics":
                self.get_client_metrics(),

            "convergence_metrics":
                self.get_convergence_metrics()
        }

        path = os.path.join(
            self.output_dir,
            "metrics.json"
        )

        with open(
            path,
            "w"
        ) as file:

            json.dump(
                data,
                file,
                indent=4
            )

        return path

    # =========================================================
    # Save global CSV
    # =========================================================

    def save_global_csv(self):

        path = os.path.join(
            self.output_dir,
            "global_metrics.csv"
        )

        with open(
            path,
            "w",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([

                "round",
                "accuracy",
                "loss",
                "client_mean_loss",
                "client_min_loss",
                "client_max_loss",
                "client_loss_std",
                "client_loss_range"

            ])

            for i in range(
                len(self.rounds)
            ):

                writer.writerow([

                    self.rounds[i],

                    self.global_accuracy[i],

                    self.global_loss[i],

                    self.client_mean_loss[i],

                    self.client_min_loss[i],

                    self.client_max_loss[i],

                    self.client_loss_std[i],

                    self.client_loss_range[i]

                ])

        return path

    # =========================================================
    # Save client CSV
    # =========================================================

    def save_client_csv(self):

        path = os.path.join(
            self.output_dir,
            "client_metrics.csv"
        )

        with open(
            path,
            "w",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([

                "round",
                "client_id",
                "training_loss"

            ])

            for client_id, losses in (
                self.client_losses.items()
            ):

                for round_number, loss in enumerate(
                    losses,
                    start=1
                ):

                    writer.writerow([

                        round_number,

                        client_id,

                        loss

                    ])

        return path

    # =========================================================
    # Save everything
    # =========================================================

    def save_all(self):

        json_path = self.save_json()

        global_csv = self.save_global_csv()

        client_csv = self.save_client_csv()

        return {

            "experiment_directory":
                self.output_dir,

            "json":
                json_path,

            "global_csv":
                global_csv,

            "client_csv":
                client_csv
        }