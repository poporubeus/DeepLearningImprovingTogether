import sys
import gc
import os
sys.path.insert(0,'/home/fv/storage1/qml/DeepLearningBaby/data')
sys.path.insert(1,'/home/fv/storage1/qml/DeepLearningBaby/')## this is the path of MY PC-change it!
import torch
from data_loading import train_loader, val_loader, test_loader
from utils import train, test
from c_VGG import CustomClassicalVGG
import json

jason_path = "/mnt/storage1/fv/qml/DeepLearningBaby/VGG16-preTrained/"

with open(os.path.join(jason_path,"config_file.json"), "r") as config_file:
    config = json.load(config_file)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

epochs = config["Epochs"]
lr = config["Learning_rate"]

if __name__ == '__main__':
    torch.cuda.empty_cache()
    gc.collect()
    gpu_dev = torch.device('cuda:0')
    print("Training classical VGG model...")
    model_save_PATH = "/home/fv/storage1/qml/DeepLearningBaby/"
    classical_model = CustomClassicalVGG()
    train(model=classical_model,
          train_loader=train_loader,
          val_loader=val_loader,
          epochs=epochs,
          learning_rate=lr,
          seed=8888,
          device=gpu_dev,
          path_to_save=model_save_PATH)

    acc, preds = test(model=classical_model, test_loader=test_loader, device=gpu_dev)
    print("Test accuracy on Cifar2:", acc)