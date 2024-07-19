import torch.nn as nn

class VGG_extractor:
    def __init__(self, model_from_extract) -> None:
        """
        Class which extracts feature and classifier sub-models upon request from user.
        """
        self.model_from_extract = model_from_extract
    def extract(self, feature_block: bool=False, classifier_block: bool=False, avg_block: bool=False) -> nn.Sequential:
        if feature_block:
            feature_extractor = nn.Sequential(*self.model_from_extract.features)
            return feature_extractor
        elif classifier_block:
            classifier_extractor = nn.Sequential(*self.model_from_extract.classifier[:-1])
            return classifier_extractor
        elif avg_block:
            avgpool_extractor = self.model_from_extract.avgpool
            return avgpool_extractor
        else:
            raise ValueError("Please, select features or classifier.")