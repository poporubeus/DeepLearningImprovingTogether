import pennylane as qml


n_qubits = 12
n_layers = 3


def FeatureMap(data):
    #qml.transforms.broadcast_expand(qml.MottonenStatePreparation(state_vector=torch.nn.functional.normalize(data), wires=range(n_qubits)))
    qml.AmplitudeEmbedding(features=data, wires=range(n_qubits), pad_with=0., normalize=True)


def QFlatten(weights, qubit):
    qml.U3(weights[0], weights[1], weights[2], wires=qubit)

def QConv(weights, qubit):
    qml.U3(weights[0], weights[1], weights[2], wires=qubit)

def QPool(weights, controlled, target_qubit):
    qml.CNOT(wires=[controlled, target_qubit])
    qml.RX(weights, wires=target_qubit)
    qml.CNOT(wires=[controlled, target_qubit])

def mid_measure(measured_qubit, target_qubit):
    m0 = qml.measure(measured_qubit)
    qml.cond(m0, qml.Identity)(wires=target_qubit)

def quantum_block(omega, qubit1, qubit2):
    QConv(omega, qubit1)
    QConv(omega, qubit2)