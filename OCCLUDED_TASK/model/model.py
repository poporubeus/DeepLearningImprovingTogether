import pennylane as qml
import torch.nn as nn
from torch import Tensor, log
from q_class import StatePreparation, Ansatz
from open_json import qbits, qdev_name


class quantum_layer(nn.Module):
    """
    Quantum model class which implements quantum neural network flow using a qnn.TorchLayer.
    The weight_shapes variable is a 3d tensor with the form (L, Q, 3). L stands for number of layers;
    while Q is the number of qubits used in the parameterized quantum circuit. 3 is fixed since
    each unitary acceptes 3 variational parameters.
    """
    def __init__(self, layers: int, qubits: int) -> None:
        super(quantum_layer, self).__init__()
        weight_shapes = {
            "weights": (layers, qubits, 3)
        }
        qlayer = qml.qnn.TorchLayer(qnn, weight_shapes)
        torch_layer = [qlayer]
        self.q_layer = nn.Sequential(*torch_layer)

    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape(x.shape[0], -1)
        x = self.q_layer(x)
        return x



class ChildreNet(nn.Module):
    """
    ChildreNet CNN which is composed by 3 main convolutional layers,
    a flatten layer, 1 fully connected layer and ends with a quantum classifier which
    outputs probabilities.
    Each of the 3 conv layers merges Conv2d, BatchNorm, ReLU and MaxPool2d with appropriate
    dimensions, kernel size, stride ecc... respectively.
    After the qnn intervention, a logarithm operation is applied to each output before
    feeding them to the Negative Log Likelihood loss.
    """
    def __init__(self, layers: int, qubits: int) -> None:
        super(ChildreNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.flatten_size = 256 * 45 * 49

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.flatten_size, 512),
            nn.ReLU()
        )
        self.quantum_classifier = quantum_layer(layers, qubits)
        #self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = out.reshape(out.shape[0], -1)
        out = self.quantum_classifier(out)
        out = log(out)
        #out = self.fc2(out)
        return out
    

qdev = qml.device(qdev_name, range(qbits))


@qml.qnode(device=qdev, interface="torch")
def qnn(inputs: Tensor, weights: Tensor) -> Tensor:
    """
    Creates the qnode and the simulated quantum circuit.
    """
    StatePreparation(inputs, qbits)
    Ansatz(weights, qubits=range(qbits))
    return qml.probs(wires=range(qbits))