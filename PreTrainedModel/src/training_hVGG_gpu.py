import sys
import gc
import os

sys.path.insert(0,'/home/fv/storage1/qml/DeepLearningBaby/data')
sys.path.insert(1,'/home/fv/storage1/qml/DeepLearningBaby/')## this is the path of MY PC-change it!
import torch
from data_loading import train_loader, val_loader, test_loader
from utils import train, test
from hybridVGG import HybridVGG, n_layers, n_qubits

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if __name__ == '__main__':
    torch.cuda.empty_cache()
    gc.collect()
    gpu_dev = torch.device('cuda:0')
    print("Training hybrid quantum model...")
    model_save_PATH = "/home/fv/storage1/qml/DeepLearningBaby/"
    hybrid_model = HybridVGG(layers=n_layers, qubits=n_qubits)
    train(model=hybrid_model, train_loader=train_loader, val_loader=val_loader, epochs=100, learning_rate=0.001, seed=8888, device=gpu_dev, path_to_save=model_save_PATH)

    acc, preds = test(model=hybrid_model, test_loader=test_loader, device=gpu_dev)
    print("Test accuracy on Cifar2:", acc)