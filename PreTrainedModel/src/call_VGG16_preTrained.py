import sys
sys.path.insert(0,'/home/fv/storage1/qml/DeepLearningBaby/data') ## this is the path of MY PC-change it!
from torchvision import models
import torch
from data.data_loading import train_loader, test_loader, val_loader
import torch.optim as optim
import torch.nn as nn
from utils import train, evaluate, test
import numpy as np


cpu_device = torch.device('cpu')
gpu_device = torch.device("cuda:0")
vgg16_pre_trained = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
vgg16_pre_trained.to(gpu_device)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = optim.Adam(vgg16_pre_trained.parameters(), lr=learning_rate)

print(np.array(train_loader.dataset.dataset.data[0]).shape)  ### shape is correct -> (224, 224, 3)
print("Train my VGG model...")

train(model=vgg16_pre_trained,
      train_loader=train_loader,
      val_loader=val_loader,
      epochs=10,
      device=gpu_device,
      learning_rate=learning_rate,
      seed=8888)


