import sys
sys.path.insert(0,'/home/fv/storage1/qml/DeepLearningBaby/data') ## this is the path of MY PC-change it!
from torchvision import models
import torch
from data.data_loading import train_loader, val_loader, test_loader
import torch.optim as optim
import torch.nn as nn
from utils import train, test, test_classification_report
import logging
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


model_save_PATH = "/home/fv/storage1/qml/DeepLearningBaby/VGG16-preTrained/RESULTS/"
csv_folder = model_save_PATH + "/csv_files/"
cpu_device = torch.device('cpu')
gpu_device = torch.device("cuda:0")
vgg16_pre_trained = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
vgg16_pre_trained.to(gpu_device)

criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = optim.Adam(vgg16_pre_trained.parameters(), lr=learning_rate)

for param in vgg16_pre_trained.features.parameters():
    param.require_grad = False

print("Modello:", vgg16_pre_trained)

num_features = vgg16_pre_trained.classifier[6].in_features
features = list(vgg16_pre_trained.classifier.children())[:-1]

feature_extractor = vgg16_pre_trained.features
'''for name, param in feature_extractor.named_parameters():
    print(name, param.data)'''

'''for img, label in train_loader:
    img = img.to(gpu_device)
    out = feature_extractor(img)
    print(out.shape)'''

#out = vgg16_pre_trained.features(train_loader.dataset.data[0])
#print(out)

'''for name, param in vgg16_pre_trained.features.named_parameters():
    print(name, param.data)'''

print("Params:", vgg16_pre_trained.classifier[0].in_features)
print("Model:", vgg16_pre_trained)

'''features.extend([nn.Linear(num_features, 2)])
vgg16_pre_trained.classifier = nn.Sequential(*features).to(device=gpu_device)


logging.info('Training only the last layer of the VGG16 (which classifies only 2 features), while keeping'
             ' the previous layers untouched...')

train(model=vgg16_pre_trained,
      train_loader=train_loader,
      val_loader=val_loader,
      epochs=5,
      device=gpu_device,
      learning_rate=learning_rate,
      seed=8888,
      path_to_save=model_save_PATH
      )


acc, preds = test(model=vgg16_pre_trained, test_loader=test_loader, device=gpu_device)
logging.info(f'Test accuracy: {acc}')


df, conf, csv_file_result = test_classification_report(model=vgg16_pre_trained,
                                                       test_loader=test_loader,
                                                       device=cpu_device,
                                                       csv_destination=model_save_PATH,
                                                       file_name="test_classification_report.csv")


'''


feature_extractor2 = nn.Sequential(*[vgg16_pre_trained.features, vgg16_pre_trained.avgpool])
print("Feature extractor 2:", feature_extractor2)

for img, label in train_loader:
    img = img.to(gpu_device)
    out = feature_extractor2(img)
    print(out.shape)