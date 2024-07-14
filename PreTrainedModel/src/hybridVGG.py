import pennylane as qml
import torch
from simple_qmodel import QNN, n_layers, n_qubits
from extractor import Extractor
from torchvision import models
import torch.nn as nn
import sys


sys.path.insert(0,'/home/fv/storage1/qml/DeepLearningBaby/PreTrainedModel/src/quantum')

gpu_device = torch.device("cuda:0")

vgg16_pre_trained = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
vgg16_pre_trained.to(gpu_device)


Net_extracted = Extractor(model_from_extract=vgg16_pre_trained)
VggNet = Net_extracted.extract(features=True, classifier=False)
print(VggNet)


class quantum_classifier(nn.Module):
    """
    Quantum model class which implements quantum neural network flow.
    """
    def __init__(self, layers: int, qubits: int) -> None:
        super(quantum_classifier, self).__init__()
        weight_shapes = {"weights": (layers, qubits, 3)}
        q_layer = qml.qnn.TorchLayer(QNN, weight_shapes)
        torch_layer = [q_layer]
        self.qlayer = torch.nn.Sequential(*torch_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        x = self.qlayer(x)
        return x


class HybridVGG(nn.Module):
    """
    A hybrid classical quantum class which implements the operations acting on data and trainable weights.
    """
    def __init__(self, layers: int, qubits: int) -> None:
        super(HybridVGG, self).__init__()
        self.features = VggNet
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.pre_classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096)
        )
        self.quantum_classifier = quantum_classifier(layers, qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.pre_classifier(x)
        outputs = self.quantum_classifier(x)
        return outputs


if __name__ == "__main__":
    model = HybridVGG(layers=n_layers, qubits=n_qubits)
    print("Hybrid model:", model)