import pennylane as qml
import matplotlib.pyplot as plt
import torch


n_qubits = 12
n_layers = 3
q_device = qml.device("default.qubit", wires=n_qubits)


def FeatureMap(data: torch.asarray) -> None:
    qml.AmplitudeEmbedding(features=data, wires=range(n_qubits), pad_with=0., normalize=True)


def Ansatz(weights: torch.asarray, qubits: list) -> None:
    qml.StronglyEntanglingLayers(weights=weights, wires=qubits)


@qml.qnode(device=q_device, interface="torch")
def QNN(inputs: torch.asarray, weights: torch.asarray) -> qml.probs:
    FeatureMap(inputs)
    Ansatz(weights, qubits=list(range(n_qubits)))
    return qml.probs(wires=1)


if __name__ == "__main__":
    features = torch.rand(16, 1, requires_grad=False)
    theta = torch.rand(n_layers, n_qubits, 3, requires_grad=False)
    qml.draw_mpl(qnode=QNN, decimals=2)(features, theta)
    plt.show()
    plt.close()