import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import os
from open_json import batch


sorgent_folder = "/Users/francescoaldoventurelli/Desktop/"
csv_file = sorgent_folder + "children_data.csv"
img_folder = sorgent_folder + "children_dataset"
batch_size = batch


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ConvertImageDtype(torch.float),  
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        '''if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])'''
    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        img_url = self.img_labels.iloc[idx, 0]
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, img_url



custom_data = CustomImageDataset(annotations_file=csv_file, img_dir=img_folder)

#train_dataloader = DataLoader(custom_data, batch_size=64, shuffle=True)
#test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


'''print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")'''

'''img = train_features[0]
label = train_labels[0]

img_permuted = img.permute(1, 2, 0)
img_np = img_permuted.numpy()'''


'''fig, axes = plt.subplots(3, 3, figsize=(9, 9))

for i, ax in enumerate(axes.flat):

    img = train_features[i]  
    img_permuted = img.permute(1, 2, 0)  
    img_np = img_permuted.numpy()

    ax.imshow(img_np)
    ax.set_title(f"Label: {train_labels[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()'''


#train_features, train_labels = next(iter(train_dataloader))

val_size = int(0.4*len(custom_data))
train_size = len(custom_data) - val_size
train_set, val_set = torch.utils.data.random_split(custom_data, [int(train_size), int(val_size)])

test_size = int(0.2*len(val_set))
val_size2 = len(val_set) - test_size

test_set, val_set2 = torch.utils.data.random_split(val_set, [val_size2, test_size])



train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)


'''print(f"Training set size: {len(train_set)}")
print(f"Validation set size: {len(val_set)}")
print(f"Test set size: {len(test_set)}")'''


def make_plot(dataset_):
    features, labels = next(iter(dataset_))
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for i, ax in enumerate(axes.flat):

        img = features[i]  
        img_permuted = img.permute(1, 2, 0)  
        img_np = img_permuted.numpy()

        ax.imshow(img_np)
        ax.set_title(f"Label: {labels[i]}")
        ax.axis('off')
    return fig


#make_plot(train_loader)
'''make_plot(test_loader)
plt.tight_layout()
plt.show()'''
        
#### 1 = NON OCCLUSO;
#### 0 = OCCLUSO.