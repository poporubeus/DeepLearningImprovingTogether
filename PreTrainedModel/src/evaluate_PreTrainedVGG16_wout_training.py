import sys
sys.path.insert(0,'/home/fv/storage1/qml/DeepLearningBaby/data') ## this is the path of MY PC-change it!
from torchvision import models
import torch
from data.data_loading import test_loader
import torch.optim as optim
import torch.nn as nn
from utils import test
import time


cpu_device = torch.device('cpu')
gpu_device = torch.device("cuda:0")
vgg16_pre_trained = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
vgg16_pre_trained.to(gpu_device)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = optim.Adam(vgg16_pre_trained.parameters(), lr=learning_rate)

'''print(np.array(train_loader.dataset.dataset.data[0]).shape)  ### shape is correct -> (224, 224, 3)
print("Train my VGG model...")'''

#print(vgg16_pre_trained.classifier[6].out_features)  ## -> 1000 -> VGG16 is able to classify 1000 distinct objects!!!

### Freeze all the parameters before the layer I want to modify (and consequentely, train)
for param in vgg16_pre_trained.features.parameters():
    param.require_grad = False

num_features = vgg16_pre_trained.classifier[6].in_features
features = list(vgg16_pre_trained.classifier.children())[:-1] # Remove last layer

features.extend([nn.Linear(num_features, 2)]) # Add our layer with 2 outputs
vgg16_pre_trained.classifier = nn.Sequential(*features).to(device=gpu_device) # Replace the model classifier
#print(vgg16_pre_trained)

start_time = time.time()
print("Test before training...")
acc, predictions = test(model=vgg16_pre_trained, test_loader=test_loader, device=gpu_device)
end_time = time.time()

print("Test accuracy without pre-training on CIFAR2:", acc)
print("Predictions:", predictions)
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time)

