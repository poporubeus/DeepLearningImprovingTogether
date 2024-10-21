import PIL.Image
import pennylane as qml
import torch.nn as nn
from torch import Tensor
from q_class import StatePreparation, Ansatz
from open_json import qbits, qdev_name, shots
from PIL import Image
import torch
import os


folder = "/Users/francescoaldoventurelli/Desktop/occluded_task/children_DATA/train/"

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



class ChildNet(nn.Module):
    def __init__(self, layers, qubits):
        super(ChildNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=1),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        '''self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=1),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )'''
        self.flatten_size = 1014

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.flatten_size, 512),
            nn.ReLU()
        )
        
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 92),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(92, 2)
        #self.quantum_classifier = quantum_layer(layers, qubits)

    def forward(self, x):
        x = self.layer1(x)
        #x = self.layer2(x)
        #x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        #x = x.reshape(x.shape[0], -1)
        #x = self.quantum_classifier(x)
        x = self.fc3(x)
        #x = torch.log(x)
        return x
    

qdev = qml.device(qdev_name, range(qbits))


@qml.qnode(device=qdev, interface="torch")
def qnn(inputs: Tensor, weights: Tensor) -> Tensor:
    """
    Creates the qnode and the simulated quantum circuit.
    """
    StatePreparation(inputs, qbits)
    Ansatz(weights, qubits=range(qbits))
    return qml.probs(wires=range(qbits))
#ciao



def get_conv_params(m_kernel, n_kernel, d_filters_previous, k_filters):
    return ((m_kernel * n_kernel * d_filters_previous) + 1) * k_filters





