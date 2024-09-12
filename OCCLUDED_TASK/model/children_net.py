import torch
import torch.nn as nn
from custom_loader import train_loader, test_loader, val_loader
from tqdm import tqdm


dimensions = (367, 398)
num_classes = 2
device = torch.device("cpu")


class ChildrenNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super(ChildrenNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output size: (64, 183, 199)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output size: (128, 91, 99)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output size: (256, 45, 49)
        )

        self.flatten_size = 256 * 45 * 49  # From (256, 45, 49)

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.flatten_size, 512),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out



model = ChildrenNet(num_classes)
criterion = nn.BCEWithLogitsLoss()

lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  

epochs = 10


for epoch in tqdm(range(epochs)):
    model.train()
    running_loss = 0.0 
    total_correct = 0
    total_instances = 0
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs[:, 1], labels.type(torch.FloatTensor))
        #loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        with torch.no_grad():
            classifications = torch.argmax(outputs, dim=1)
            correct_predictions = (classifications == labels).sum().item()
            total_correct += correct_predictions
            total_instances += len(images)
    # Validation
    '''model.eval()
    val_loss = 0.0
    total_val_correct = 0
    total_val_instances = 0

    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            v_outputs = model(val_inputs)
            loss = criterion(v_outputs[:, 1], val_labels.type(torch.FloatTensor))
            val_loss += loss.item()

            classifications = torch.argmax(v_outputs, dim=1)
            total_val_correct += (classifications == val_labels).sum().item()
            total_val_correct += len(val_inputs)'''
    print(
            f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}, "
            f"Accuracy: {(total_correct / total_instances) * 100:.2f}%, "
            #f"Val loss: {val_loss / len(val_loader)}, "
            #f"Val accuracy: {(total_val_correct / total_val_instances) * 100:.2f}"
        )