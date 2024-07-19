import pennylane as qml
import torch
from simple_qmodel import QNN
from vgg_extractor import VGG_extractor
from torchvision import models
import torch.nn as nn
import sys


sys.path.insert(0,'/home/fv/storage1/qml/DeepLearningBaby/PreTrainedModel/src/quantum')

gpu_device = torch.device("cuda:0")

vgg16_pre_trained = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
vgg16_pre_trained.to(gpu_device)


Net_extracted = VGG_extractor(model_from_extract=vgg16_pre_trained)
conv_block = Net_extracted.extract(feature_block=True, classifier_block=False, avg_block=False)
avg_block = Net_extracted.extract(feature_block=False, classifier_block=False, avg_block=True)
pre_classifier_block = Net_extracted.extract(feature_block=False, classifier_block=True, avg_block=False)


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
        self.features = conv_block
        self.avgpool = avg_block
        self.pre_classifier = pre_classifier_block
        self.quantum_classifier = quantum_classifier(layers, qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.pre_classifier(x)
        outputs = self.quantum_classifier(x)
        return outputs