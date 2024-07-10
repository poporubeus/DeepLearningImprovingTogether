import sys
sys.path.insert(0,'/home/fv/storage1/qml/DeepLearningBaby/data') ## this is the path of MY PC-change it!
from torchvision import models
import torch
import torch.optim as optim
import torch.nn as nn
import logging


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


### Frizzare i pesi di imagenet -> else: commentare queste linee
for param in vgg16_pre_trained.features.parameters():
    param.require_grad = False


class Extractor:
    def __init__(self, model_from_extract) -> None:
        self.model_from_extract = model_from_extract
        """
        Class which extracts feature and classifier sub-models upon request from user.
        """
    def extract(self, features: bool=False, classifier: bool=False) -> nn.Sequential:
        if features and classifier:
            raise ValueError("Please, select either features or classifier, not both.")
        if features:
            feature_extractor = nn.Sequential(*self.model_from_extract.features)
            return feature_extractor
        elif classifier:
            classifier_extractor = nn.Sequential(*self.model_from_extract.classifier)
            return classifier_extractor
        else:
            raise ValueError("Please, select features or classifier.")


extraction = Extractor(model_from_extract=vgg16_pre_trained)
print("Feature extractor:", extraction.extract(features=False, classifier=True))