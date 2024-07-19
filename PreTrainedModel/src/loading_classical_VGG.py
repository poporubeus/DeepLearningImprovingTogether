import torch
from train_with_different_parts import train, evaluate
from data_loading import train_loader, val_loader, test_loader
import json
import os
from hybridVGG import quantum_classifier
from simple_qmodel import n_layers, n_qubits
from training_cVGG import model_name
from c_VGG import CustomClassicalVGG
import torch.optim as optim


jason_path = "/mnt/storage1/fv/qml/DeepLearningBaby/VGG16-preTrained/"
model_save_PATH = "/mnt/storage1/fv/qml/DeepLearningBaby/VGG16-preTrained/RESULTS/"

with open(os.path.join(jason_path,"config_file.json"), "r") as config_file:
    config = json.load(config_file)


epochs = config["Epochs"]
lr = config["Learning_rate"]
seed = config["Seed"]

gpu_dev = torch.device("cuda:0")
classical_VGG_model = CustomClassicalVGG()
classical_model_name = model_name
quantum_model_name = "q_layer_alone"
checkpoint = torch.load(model_save_PATH + classical_model_name + "_log.pt")
classical_VGG_model.load_state_dict(checkpoint['model_state_dict'])


for param in classical_VGG_model.parameters():
    param.requires_grad = False


# Sostituisco al classifier il quantum layer
q_layer = quantum_classifier(layers=n_layers, qubits=n_qubits)
classical_VGG_model.classifier = q_layer
q_optimizer = optim.Adam(classical_VGG_model.classifier.parameters(), lr=lr)


if __name__ == "__main__":
    train(model=classical_VGG_model,
          train_loader=train_loader,
          val_loader=val_loader,
          epochs=epochs,
          learning_rate=lr,
          seed=seed,
          device=gpu_dev,
          path_to_save=model_save_PATH,
          model_name=quantum_model_name,
          optimizer=None)