import os
import json


# Open configuration json file
jason_path = "/Users/francescoaldoventurelli/Desktop/"

with open(os.path.join(jason_path,"config.json"), "r") as config_file:
    config = json.load(config_file)


qbits = config["Qubits"]
qdev_name = config["Qdev"]
seed = config["Seed"]
layers = config["Layers"]
lr = config["Learning Rate"]
epochs = config["Epochs"]
batch = config["Batch Size"]