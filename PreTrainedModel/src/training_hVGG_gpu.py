import sys
import gc
import os
sys.path.insert(0,'/home/fv/storage1/qml/DeepLearningBaby/data')
sys.path.insert(1,'/home/fv/storage1/qml/DeepLearningBaby/')## this is the path of MY PC-change it!
import torch
from data_loading import train_loader, val_loader, test_loader
from utils import test
from train_with_different_parts import train
from hybridVGG import HybridVGG
from simple_qmodel import n_layers, n_qubits
import json


jason_path = "/mnt/storage1/fv/qml/DeepLearningBaby/VGG16-preTrained/"

with open(os.path.join(jason_path,"config_file.json"), "r") as config_file:
    config = json.load(config_file)


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

epochs = config["Epochs"]
lr = config["Learning_rate"]
seed = config["Seed"]

if __name__ == '__main__':
    torch.cuda.empty_cache()
    gc.collect()
    gpu_dev = torch.device('cuda:0')
    print("Training hybrid quantum model...")
    model_save_PATH = "/home/fv/storage1/qml/DeepLearningBaby/VGG16-preTrained/RESULTS"
    model_name = "VGG16-preTrained_w_Qlayer"
    hybrid_model = HybridVGG(layers=n_layers, qubits=n_qubits)
    train(model=hybrid_model,
          train_loader=train_loader,
          val_loader=val_loader,
          epochs=epochs,
          learning_rate=lr,
          seed=seed,
          device=gpu_dev,
          path_to_save=model_save_PATH,
          model_name=model_name,
          optimizer=None)

    acc, preds = test(model=hybrid_model, test_loader=test_loader, device=gpu_dev)
    print("Test accuracy on Cifar2:", acc)