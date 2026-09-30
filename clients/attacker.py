from copy import deepcopy

from clients.client import Client


class Attacker(Client):

    def __init__(
        self,
        train_loader,
        learning_rate=0.01,
        attack_type="none",
        attack_strength=1.0
    ):

        super().__init__(
            train_loader=train_loader,
            learning_rate=learning_rate
        )

        # =====================================================
        # Attack configuration
        # =====================================================

        self.attack_type = attack_type
        self.attack_strength = attack_strength

    # =========================================================
    # Generate client model
    # =========================================================

    def get_weights(self):

        local_weights = super().get_weights()

        if self.attack_type == "none":
            return local_weights

        update = self.calculate_update(
            local_weights
        )

        malicious_update = self.apply_attack(
            update
        )

        return self.apply_update(
            malicious_update
        )

    # =========================================================
    # Calculate local model update
    # =========================================================

    def calculate_update(self, local_weights):

        if self.global_weights is None:

            raise RuntimeError(
                "Global weights are not available. "
                "set_weights() must be called before "
                "generating an attacker update."
            )

        update = {}

        for name in local_weights:

            update[name] = (
                local_weights[name]
                - self.global_weights[name]
            )

        return update

    # =========================================================
    # Attack dispatcher
    # =========================================================

    def apply_attack(self, update):

        if self.attack_type == "sign_flip":

            return self.sign_flip(update)

        raise ValueError(
            f"Unsupported attack type: "
            f"{self.attack_type}"
        )

    # =========================================================
    # Sign-flipping attack
    # =========================================================

    def sign_flip(self, update):

        malicious_update = {}

        for name in update:

            malicious_update[name] = (
                -self.attack_strength
                * update[name]
            )

        return malicious_update

    # =========================================================
    # Convert update back into model weights
    # =========================================================

    def apply_update(self, update):

        malicious_weights = deepcopy(
            self.global_weights
        )

        for name in update:

            malicious_weights[name] = (
                self.global_weights[name]
                + update[name]
            )

        return malicious_weights