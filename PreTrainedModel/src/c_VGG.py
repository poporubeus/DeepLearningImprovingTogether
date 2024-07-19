import torch
from torchvision import models
import torch.nn as nn
from vgg_extractor import VGG_extractor


gpu_device = torch.device("cuda:0")
vgg16_pre_trained = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
vgg16_pre_trained.to(gpu_device)

extractor = VGG_extractor(model_from_extract=vgg16_pre_trained)
conv_block_VGG = extractor.extract(feature_block=True, classifier_block=False, avg_block=False)
avgpool_block_VGG = extractor.extract(feature_block=False, classifier_block=False, avg_block=True)
pre_classifier_block_VGG = extractor.extract(feature_block=False, classifier_block=True, avg_block=False)


class CustomClassicalVGG(nn.Module):
    """
    This class creates a custom VGG that does not differ from the actual VGG architecture.
    After downloading the features block, an Adaptive AvgPooling layer has been applied to resize
    everything to get 25088 features at the end. A pre classifier is created to scale 25088 features
    into 4096 output features which are subsequently used to train the quantum neural network placed at the
    end of the network. In this case, since the VGG is classical, we attach a classifier block which makes the
    classification of the remaining two distinct classes of the dataset. In particular, it brings the 4096
    features into 2 and computes the probability of getting the class 0 ora the class 1.
    Once we train the classical model, the idea will be to extract tensors of dimension
    (BATCH_SIZE, 4096) and to use it as a new dataset to train only the quantum neural network layer.
    """
    def __init__(self) -> None:
        super(CustomClassicalVGG, self).__init__()
        self.features = conv_block_VGG
        self.avgpool = avgpool_block_VGG
        self.pre_classifier = pre_classifier_block_VGG
        self.classifier = nn.Linear(in_features=4096, out_features=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.pre_classifier(x)
        outputs = self.classifier(x)
        return outputs