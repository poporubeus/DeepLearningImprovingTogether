import pennylane as qml
import matplotlib.pyplot as plt
import torch

n_qubits = 12
n_layers = 3
device = qml.device("default.qubit", wires=n_qubits)


def FeatureMap(data):
    qml.AmplitudeEmbedding(features=data, wires=range(n_qubits), pad_with=0., normalize=True)


def Ansatz(weights: torch.asarray, qubits: list):
    qml.StronglyEntanglingLayers(weights=weights, wires=qubits)


def Ansatz_w_measurement(weights, qubits: list):
    qml.StronglyEntanglingLayers(weights=weights, wires=qubits)
    m0 = qml.measure(0)
    m2 = qml.measure(2)
    m4 = qml.measure(4)
    m6 = qml.measure(6)
    m8 = qml.measure(8)
    m10 = qml.measure(10)
    qml.cond(m0, qml.Identity)(wires=1)
    qml.cond(m2, qml.Identity)(wires=3)
    qml.cond(m4, qml.Identity)(wires=5)
    qml.cond(m6, qml.Identity)(wires=7)
    qml.cond(m8, qml.Identity)(wires=9)
    qml.cond(m10, qml.Identity)(wires=11)


@qml.qnode(device, interface="torch")
def QNN(data, w, v):
    FeatureMap(data)
    Ansatz_w_measurement(w, qubits=list(range(n_qubits)))
    Ansatz(v, qubits=[j for j in range(1, 11, 2)])
    return qml.probs(wires=1)


if __name__ == "__main__":
    features = torch.rand(16, 1, requires_grad=False)
    theta = torch.rand(n_layers, n_qubits, 3, requires_grad=False)
    omega = torch.rand(n_layers, 5, 3, requires_grad=False)
    qml.draw_mpl(qnode=QNN, decimals=2)(features, theta, omega)
    plt.show()