import torch
from torchvision import datasets, transforms
import ssl
import numpy as np
from upsampling import UpSample


torch.manual_seed(46)
s = 224 ### size of new images
ssl._create_default_https_context = ssl._create_unverified_context
data_folder = "/home/fv/storage1/qml/cifar_data"

transforms_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

FULL_dataset = datasets.CIFAR10(root=data_folder, train=True, download=False, transform=transforms_cifar)
test_dataset = datasets.CIFAR10(root=data_folder, train=False, download=False, transform=transforms_cifar)

train01_idx = np.where(np.array(FULL_dataset.targets) == 0 | (np.array(FULL_dataset.targets) == 1))[0]
test01_idx = np.where(np.array(test_dataset.targets) == 0 | (np.array(test_dataset.targets) == 1))[0]

FULL_dataset.data = FULL_dataset.data[train01_idx]
FULL_dataset.targets = np.array(FULL_dataset.targets)[train01_idx].tolist()
test_dataset.data = test_dataset.data[test01_idx]
test_dataset.targets = np.array(test_dataset.targets)[test01_idx].tolist()


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, target


upsampled_full_set = [UpSample(image, s) for image in FULL_dataset.data]
upsampled_test_set = [UpSample(image, s) for image in test_dataset.data]


new_train_dataset = CustomDataset(upsampled_full_set, FULL_dataset.targets, transform=transforms_cifar)
new_test_dataset = CustomDataset(upsampled_test_set, test_dataset.targets, transform=transforms_cifar)


val_size = 0.3*len(new_train_dataset)
train_size = len(new_train_dataset) - val_size
train_set, val_set = torch.utils.data.random_split(new_train_dataset, [int(train_size), int(val_size)])

batch_size = 128
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(new_test_dataset, batch_size=batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)