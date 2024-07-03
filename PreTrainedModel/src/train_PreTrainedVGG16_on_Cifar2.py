import sys
sys.path.insert(0,'/home/fv/storage1/qml/DeepLearningBaby/data') ## this is the path of MY PC-change it!
from torchvision import models
import torch
from data.data_loading import train_loader, val_loader, test_loader
import torch.optim as optim
import torch.nn as nn
from utils import train, test
import logging


cpu_device = torch.device('cpu')
gpu_device = torch.device("cuda:0")
vgg16_pre_trained = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
vgg16_pre_trained.to(gpu_device)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = optim.Adam(vgg16_pre_trained.parameters(), lr=learning_rate)

for param in vgg16_pre_trained.features.parameters():
    param.require_grad = False

num_features = vgg16_pre_trained.classifier[6].in_features
features = list(vgg16_pre_trained.classifier.children())[:-1]
features.extend([nn.Linear(num_features, 2)])
vgg16_pre_trained.classifier = nn.Sequential(*features).to(device=gpu_device)

logging.info('Training only the last layer of the VGG16 (which classifies only 2 features), while keeping'
             ' the previous layers untouched...')

train(model=vgg16_pre_trained,
      train_loader=train_loader,
      val_loader=val_loader,
      epochs=10,
      device=gpu_device,
      learning_rate=learning_rate,
      seed=8888)

acc, preds = test(model=vgg16_pre_trained, test_loader=test_loader, device=gpu_device)
logging.info('Test accuracy', acc)

