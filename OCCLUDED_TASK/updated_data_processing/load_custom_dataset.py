import torch
from torchvision import datasets, transforms

path= "/Users/francescoaldoventurelli/Desktop/PAZIENTI_CROPPED"

totensor = transforms.Compose([
    transforms.ToTensor()
])  ### since Pytorch wants tensorss

train_data = datasets.ImageFolder(root=path + '/train', transform=totensor)
val_data = datasets.ImageFolder(root=path + '/val', transform=totensor)
test_data = datasets.ImageFolder(root=path + '/test', transform=totensor)


train_ld = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, num_workers = 4)
val_ld = torch.utils.data.DataLoader(val_data, batch_size=16, shuffle=True, num_workers = 4)
test_ld = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False, num_workers = 4)