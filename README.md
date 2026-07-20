# Byzantine Fault Tolerant Federated Learning Experimentation Platform

A research experimentation platform for testing and evaluating Byzantine Fault Tolerance (BFT) techniques in Federated Learning (FL) systems.

The goal of this project is to create an environment where different Byzantine attacks and defense mechanisms can be implemented and compared under controlled federated learning scenarios.

---

## Overview

Federated Learning enables multiple clients to collaboratively train a machine learning model without sharing their raw data.

However, in a distributed environment, some clients may behave maliciously and send incorrect model updates. These Byzantine clients can reduce the performance or completely disrupt the training process.

This project focuses on experimenting with different approaches to make Federated Learning systems more robust against Byzantine behavior.

---

## Current Features

- Federated Learning simulation
- Multiple client training
- Centralized FL server
- Byzantine client simulation
- Testing different aggregation strategies
- Model performance evaluation

---

## Project Structure

```
BFT-FL-Experimentation/

├── client/
│   └── client.py

├── server/
│   └── server.py

├── models/
│   └── model.py

├── datasets/
│   └── data_manager.py

├── experiments/
│   └── experiments.py

├── metrics/
│   └── metrics.py

├── main.py
└── README.md
```

---

## How It Works

The system consists of:

### Server

The server:

- Maintains the global model
- Sends the model to clients
- Receives client updates
- Aggregates updates
- Updates the global model


### Clients

Each client:

- Receives the global model
- Trains locally using its own dataset
- Sends model updates back to the server


### Byzantine Clients

Byzantine clients simulate malicious behavior by sending incorrect model updates.

Examples:

- Random updates
- Modified gradients
- Model poisoning attacks

---

## Experiment Flow

```
Initialize Global Model

        |
        v

Create Clients

        |
        v

Local Training

        |
        v

Client Model Updates

        |
        v

Aggregation

        |
        v

Global Model Update

        |
        v

Evaluation
```

---

## Technologies Used

- Python
- PyTorch
- TorchVision
- Federated Learning concepts
- Byzantine Fault Tolerance concepts

---

## Future Improvements

Planned improvements:

- Add more Byzantine attack strategies
- Implement different BFT aggregation algorithms
- Add experiment configuration files
- Add visualization dashboard
- Support larger distributed simulations

---

## Purpose

This project is built as a research experimentation platform to understand and improve the reliability of Federated Learning systems under Byzantine failures.
