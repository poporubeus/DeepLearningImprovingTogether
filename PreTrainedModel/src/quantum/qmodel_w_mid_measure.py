from quantum_utils import *


device = qml.device("default.qubit.torch", wires=n_qubits, torch_device='cuda:0')


@qml.qnode(device=device, interface="torch")
def QCNN(inputs, conv_weights_1,
         pool_weights_1, pool_weights_2,
         conv_weights_2,
         pool_weights_3,
         flatten_weights_1):
    FeatureMap(inputs)
    qml.Barrier(only_visual=True)

    for i, j in enumerate(range(0, n_qubits-1, 2)):
        quantum_block(conv_weights_1[i, :], qubit1=j, qubit2=j + 1)
    qml.Barrier(only_visual=True)

    for i, j in enumerate(range(0, n_qubits-1, 2)):
        QPool(pool_weights_1[i], controlled=j, target_qubit=j+1)
    for i, j in enumerate(range(1, n_qubits-1, 2)):
        QPool(pool_weights_2[i], controlled=j, target_qubit=j+1)
    qml.Barrier(only_visual=True)

    for i, j in enumerate(range(0, n_qubits-1, 2)):
        quantum_block(conv_weights_2[i, :], qubit1=j, qubit2=j + 1)
    qml.Barrier(only_visual=True)

    for k in range(0, n_qubits-1, 2):
        mid_measure(measured_qubit=k, target_qubit=k+1)
    qml.Barrier(only_visual=True)

    for i, j in enumerate(range(0, n_qubits-2, 4)):
        quantum_block(conv_weights_2[i, :], qubit1=j+1, qubit2=j + 3)
    qml.Barrier(only_visual=True)

    for i, j in enumerate(range(1, n_qubits - 2, 2)):
        QPool(pool_weights_3[i], controlled=j, target_qubit=j + 2)
    qml.Barrier(only_visual=True)

    for i, j in enumerate(range(0, n_qubits-2, 4)):
        quantum_block(conv_weights_2[i, :], qubit1=j+1, qubit2=j + 3)
    qml.Barrier(only_visual=True)

    for k in range(0, n_qubits-2, 4):
        mid_measure(measured_qubit=k+1, target_qubit=k+3)

    qml.Barrier(only_visual=True)
    QFlatten(weights=flatten_weights_1[:, 0], qubit=3)
    QFlatten(weights=flatten_weights_1[:, 1], qubit=7)
    QFlatten(weights=flatten_weights_1[:, 2], qubit=11)
    return qml.probs(wires=[3, 7, 11])
