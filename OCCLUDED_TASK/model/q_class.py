import pennylane as qml
from torch import Tensor
    
        
def StatePreparation(features: Tensor, qubits: int):
    qml.AmplitudeEmbedding(features=features, wires=range(qubits), pad_with=0, normalize=True)

def Ansatz(weights: Tensor, qubits: int):
    qml.StronglyEntanglingLayers(weights=weights, wires=list(qubits))