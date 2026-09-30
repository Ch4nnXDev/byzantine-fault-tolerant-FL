from config.config_loader import load_config


def main():
    config = load_config("config/baseline.yaml")

    print("========================================")
    print("Experiment:", config["experiment"]["name"])
    print("Seed:", config["experiment"]["seed"])
    print("========================================")

    print("\nFederated Learning")
    print("Clients:", config["federated_learning"]["num_clients"])
    print("Rounds:", config["federated_learning"]["rounds"])
    print("Local epochs:", config["federated_learning"]["local_epochs"])
    print("Learning rate:", config["federated_learning"]["learning_rate"])

    print("\nByzantine")
    print("Enabled:", config["byzantine"]["enabled"])
    print("Nodes:", config["byzantine"]["num_nodes"])
    print("Attack:", config["byzantine"]["attack"])
    print("Attack strength:", config["byzantine"]["attack_strength"])

    print("\nTopology")
    print("Type:", config["topology"]["type"])

    print("\nDefense")
    print("Enabled:", config["defense"]["enabled"])
    print("Method:", config["defense"]["method"])


if __name__ == "__main__":
    main()