
import csv
import json
import os
import statistics
import time

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

        # Change in global accuracy from previous round
        self.accuracy_change = []

        # Change in global loss from previous round
        self.loss_change = []

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
        # Round timing
        # =====================================================

        self.round_durations = []

        self.round_start_time = None

        self.experiment_start_time = None

        self.experiment_duration = None

        # =====================================================
        # Experiment metadata
        # =====================================================

        self.metadata = {}

    # =========================================================
    # Experiment timing
    # =========================================================

    def start_experiment_timer(self):

        self.experiment_start_time = time.perf_counter()

    def start_round_timer(self):

        self.round_start_time = time.perf_counter()

    def finish_round_timer(self):

        if self.round_start_time is None:

            return 0.0

        duration = (
            time.perf_counter()
            -
            self.round_start_time
        )

        self.round_durations.append(
            duration
        )

        self.round_start_time = None

        return duration

    def finish_experiment_timer(self):

        if self.experiment_start_time is None:

            return 0.0

        self.experiment_duration = (
            time.perf_counter()
            -
            self.experiment_start_time
        )

        return self.experiment_duration

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
        seed=None,
        batch_size=None,
        learning_rate=None,
        partition_type=None,
        topology=None,
        defense_enabled=False,
        defense_method="none",
        attack="none"
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

            "batch_size":
                batch_size,

            "learning_rate":
                learning_rate,

            "partition_type":
                partition_type,

            "byzantine_clients":
                byzantine_clients,

            "attack":
                attack,

            "aggregation_method":
                aggregation_method,

            "topology":
                topology,

            "defense_enabled":
                defense_enabled,

            "defense_method":
                defense_method,

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
            float(loss)
        )

    # =========================================================
    # Calculate client statistics
    # =========================================================

    def calculate_client_statistics(
        self,
        round_number
    ):

        current_losses = []

        for losses in self.client_losses.values():

            if len(losses) >= round_number:

                current_losses.append(
                    losses[round_number - 1]
                )

        if not current_losses:

            return {

                "mean": 0.0,

                "min": 0.0,

                "max": 0.0,

                "std": 0.0,

                "range": 0.0
            }

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
            max_loss -
            min_loss
        )

        if len(current_losses) > 1:

            std_loss = statistics.stdev(
                current_losses
            )

        else:

            std_loss = 0.0

        return {

            "mean":
                mean_loss,

            "min":
                min_loss,

            "max":
                max_loss,

            "std":
                std_loss,

            "range":
                loss_range
        }

    # =========================================================
    # Record global metrics
    # =========================================================

    def record_global(
        self,
        round_number,
        accuracy,
        loss
    ):

        accuracy = float(
            accuracy
        )

        loss = float(
            loss
        )

        self.rounds.append(
            round_number
        )

        self.global_accuracy.append(
            accuracy
        )

        self.global_loss.append(
            loss
        )

        # =====================================================
        # Round-to-round changes
        # =====================================================

        if len(self.global_accuracy) > 1:

            previous_accuracy = (
                self.global_accuracy[-2]
            )

            previous_loss = (
                self.global_loss[-2]
            )

            accuracy_delta = (
                accuracy -
                previous_accuracy
            )

            loss_delta = (
                loss -
                previous_loss
            )

        else:

            # No previous round exists
            accuracy_delta = 0.0

            loss_delta = 0.0

        self.accuracy_change.append(
            accuracy_delta
        )

        self.loss_change.append(
            loss_delta
        )

        # =====================================================
        # Client statistics
        # =====================================================

        statistics_data = (
            self.calculate_client_statistics(
                round_number
            )
        )

        self.client_mean_loss.append(
            statistics_data["mean"]
        )

        self.client_min_loss.append(
            statistics_data["min"]
        )

        self.client_max_loss.append(
            statistics_data["max"]
        )

        self.client_loss_std.append(
            statistics_data["std"]
        )

        self.client_loss_range.append(
            statistics_data["range"]
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

            "accuracy_change":
                self.accuracy_change,

            "loss_change":
                self.loss_change,

            "client_mean_loss":
                self.client_mean_loss,

            "client_min_loss":
                self.client_min_loss,

            "client_max_loss":
                self.client_max_loss,

            "client_loss_std":
                self.client_loss_std,

            "client_loss_range":
                self.client_loss_range,

            "round_durations":
                self.round_durations
        }

    # =========================================================
    # Get client metrics
    # =========================================================

    def get_client_metrics(self):

        return self.client_losses

    # =========================================================
    # Client convergence analysis
    # =========================================================

    def get_client_convergence_metrics(self):

        results = {}

        for client_id, losses in (
            self.client_losses.items()
        ):

            if not losses:

                continue

            initial_loss = losses[0]

            final_loss = losses[-1]

            loss_change = (
                final_loss -
                initial_loss
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

            results[client_id] = {

                "rounds":
                    len(losses),

                "initial_loss":
                    initial_loss,

                "final_loss":
                    final_loss,

                "loss_change":
                    loss_change,

                "loss_reduction":
                    loss_reduction,

                "loss_reduction_percent":
                    loss_reduction_percent
            }

        return results

    # =========================================================
    # Calculate convergence statistics
    # =========================================================

    def get_convergence_metrics(self):

        if not self.global_accuracy:

            return {}

        # =====================================================
        # Initial / final values
        # =====================================================

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

        # =====================================================
        # Overall changes
        # =====================================================

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

        # =====================================================
        # Best accuracy
        # =====================================================

        best_accuracy = max(
            self.global_accuracy
        )

        best_accuracy_index = (
            self.global_accuracy.index(
                best_accuracy
            )
        )

        best_accuracy_round = (
            self.rounds[
                best_accuracy_index
            ]
        )

        # =====================================================
        # Accuracy improvements
        # =====================================================

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

        # =====================================================
        # Accuracy stability
        # =====================================================

        if accuracy_improvements:

            absolute_accuracy_changes = [

                abs(
                    change
                )

                for change
                in accuracy_improvements
            ]

            average_accuracy_change = (
                statistics.mean(
                    absolute_accuracy_changes
                )
            )

            if len(
                absolute_accuracy_changes
            ) > 1:

                accuracy_volatility = (
                    statistics.stdev(
                        absolute_accuracy_changes
                    )
                )

            else:

                accuracy_volatility = 0.0

        else:

            average_accuracy_change = 0.0

            accuracy_volatility = 0.0

        # =====================================================
        # Loss stability
        # =====================================================

        if len(self.global_loss) > 1:

            loss_changes = [

                self.global_loss[i]
                -
                self.global_loss[i - 1]

                for i in range(
                    1,
                    len(self.global_loss)
                )
            ]

            average_loss_change = (
                statistics.mean(
                    loss_changes
                )
            )

            absolute_loss_changes = [

                abs(
                    change
                )

                for change
                in loss_changes
            ]

            if len(
                absolute_loss_changes
            ) > 1:

                loss_volatility = (
                    statistics.stdev(
                        absolute_loss_changes
                    )
                )

            else:

                loss_volatility = 0.0

        else:

            average_loss_change = 0.0

            loss_volatility = 0.0

        # =====================================================
        # Round timing
        # =====================================================

        if self.round_durations:

            average_round_duration = (
                statistics.mean(
                    self.round_durations
                )
            )

            min_round_duration = min(
                self.round_durations
            )

            max_round_duration = max(
                self.round_durations
            )

        else:

            average_round_duration = 0.0

            min_round_duration = 0.0

            max_round_duration = 0.0

        # =====================================================
        # Return convergence metrics
        # =====================================================

        return {

            "initial_accuracy":
                initial_accuracy,

            "final_accuracy":
                final_accuracy,

            "best_accuracy":
                best_accuracy,

            "best_accuracy_round":
                best_accuracy_round,

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
                average_accuracy_change,

            "accuracy_volatility":
                accuracy_volatility,

            "average_loss_change":
                average_loss_change,

            "loss_volatility":
                loss_volatility,

            "average_round_duration":
                average_round_duration,

            "min_round_duration":
                min_round_duration,

            "max_round_duration":
                max_round_duration,

            "total_experiment_duration":
                self.experiment_duration
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

            "client_convergence":
                self.get_client_convergence_metrics(),

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

            writer = csv.writer(
                file
            )

            writer.writerow([

                "round",
                "accuracy",
                "accuracy_change",
                "loss",
                "loss_change",
                "client_mean_loss",
                "client_min_loss",
                "client_max_loss",
                "client_loss_std",
                "client_loss_range",
                "round_duration"

            ])

            for i in range(
                len(self.rounds)
            ):

                if i < len(
                    self.round_durations
                ):

                    round_duration = (
                        self.round_durations[i]
                    )

                else:

                    round_duration = 0.0

                writer.writerow([

                    self.rounds[i],

                    self.global_accuracy[i],

                    self.accuracy_change[i],

                    self.global_loss[i],

                    self.loss_change[i],

                    self.client_mean_loss[i],

                    self.client_min_loss[i],

                    self.client_max_loss[i],

                    self.client_loss_std[i],

                    self.client_loss_range[i],

                    round_duration

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

            writer = csv.writer(
                file
            )

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
    # Save round timing CSV
    # =========================================================

    def save_timing_csv(self):

        path = os.path.join(
            self.output_dir,
            "round_timing.csv"
        )

        with open(
            path,
            "w",
            newline=""
        ) as file:

            writer = csv.writer(
                file
            )

            writer.writerow([

                "round",
                "duration_seconds"

            ])

            for round_number, duration in zip(
                self.rounds,
                self.round_durations
            ):

                writer.writerow([

                    round_number,

                    duration

                ])

        return path

    # =========================================================
    # Save everything
    # =========================================================

    def save_all(self):

        # Make sure experiment timing is captured
        if (
            self.experiment_start_time is not None
            and
            self.experiment_duration is None
        ):

            self.finish_experiment_timer()

        json_path = (
            self.save_json()
        )

        global_csv = (
            self.save_global_csv()
        )

        client_csv = (
            self.save_client_csv()
        )

        timing_csv = (
            self.save_timing_csv()
        )

        return {

            "experiment_directory":
                self.output_dir,

            "json":
                json_path,

            "global_csv":
                global_csv,

            "client_csv":
                client_csv,

            "timing_csv":
                timing_csv
        }

