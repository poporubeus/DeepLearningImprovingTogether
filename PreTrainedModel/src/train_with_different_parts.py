import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
sys.path.insert(0,'/home/fv/storage1/qml/DeepLearningBaby/')
from utils import evaluate


def train(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader,
          val_loader: torch.utils.data.DataLoader, epochs: int, learning_rate: float,
          seed: int, device: torch.device, path_to_save: str, model_name: str,
          optimizer: torch.optim) -> None:
    """
    Function which trains the model on training dataloader and evaluates it
    on the validation dataloader. During training, the model is set to train mode,
    it computes loss and accuracy after each epoch either on training and validation sets.
    :param model: (torch.nn.Module) the model to be trained and validated;
    :param train_loader: (torch.utils.data.DataLoader) training data loader;
    :param val_loader: (torch.utils.data.DataLoader) validation data loader;
    :param epochs: (int) number of epochs to train the model;
    :param learning_rate: (float) learning rate;
    :param seed: (int) random seed for reproducibility;
    :param device: (torch.device) device to use the model with - it can be cpu or gpu (cuda);
    :return: None.
    """
    torch.manual_seed(seed)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0 # la loss che calcoliamo è una loss che si aggiorna dopo ogni epoca, non è dopo ogni batch.
        total_correct = 0
        total_instances = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            with torch.no_grad():
                classifications = torch.argmax(outputs, dim=1)
                correct_predictions = (classifications == labels).sum().item()
                total_correct += correct_predictions
                total_instances += len(inputs)

        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss,
        }, path_to_save + str(model_name) + "_log.pt")

        print(
            f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}, "
            f"Validation Loss: {val_loss}, "
            f"Accuracy: {(total_correct / total_instances) * 100:.2f}%, "
            f"Validation Accuracy: {val_accuracy:.2f}%"
        )
