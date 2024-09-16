from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from open_json import seed
import csv


criterion = nn.NLLLoss()
#criterion = nn.BCEWithLogitsLoss()


def convert_to_logits(prob: torch.Tensor) -> torch.Tensor:
    """
    In the case we want to use a BCE loss we need to feed it with logits.
    This function takes raw probabilities as qnn's output and convert them to logits.
    """
    eps = 1e-7 
    prob = torch.clamp(prob, min=eps, max=1 - eps)
    return torch.log(prob / (1 - prob))


def train_and_validate(model: callable,
                       epochs: int,
                       train_loader: DataLoader,
                       val_loader: DataLoader,
                       device: torch.device,
                       optimizer: torch.optim,
                       ) -> None:
    """
    Train and validate the model on the train and validation dataloader.
    :param model: (callable) the model to train and validate;
    :param epochs: (int) number of epochs;
    :param train_loader: (torch.utils.data.DataLoader) the train dataloader;
    :param val_loader: (torch.utils.data.DataLoader) the validation dataloader;
    :param device: (torch.device) the device to use for evaluation (cpu/cuda) needs to be set;
    :param optimizer: (torch.optim) the optimizer;

    :return: None.
    """
    
    torch.manual_seed(seed)
    
    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0 
        total_correct = 0
        total_instances = 0
        for i, (images, labels, _) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            #logit_output = convert_to_logits(outputs)
            #loss = criterion(logit_output[:, 1], labels.type(torch.FloatTensor))
            loss = criterion(outputs, labels)
            
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
        model.eval()
        val_loss = 0.0
        total_val_correct = 0
        total_val_instances = 0

        with torch.no_grad():
            for (val_inputs, val_labels, _) in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                v_outputs = model(val_inputs)
                #val_logit_output = convert_to_logits(v_outputs)
                #loss = criterion(val_logit_output[:, 1], val_labels.type(torch.FloatTensor))
                loss = criterion(v_outputs, val_labels)
                val_loss += loss.item()

                classifications = torch.argmax(v_outputs, dim=1)
                total_val_correct += (classifications == val_labels).sum().item()
                total_val_instances += len(val_inputs)

        ### save model's logs
        if epoch % 5 == 0:
            torch.save(
                {'epochs': epoch + 1, 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(), 'training_loss': running_loss},
                f"weights.pt")
        print(
                f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}, "
                f"Accuracy: {(total_correct / total_instances) * 100:.2f}%, "
                f"Val loss: {val_loss / len(val_loader)}, "
                f"Val accuracy: {(total_val_correct / total_val_instances) * 100:.2f}%"
            )
    

def test(model: torch.nn.Module, test_loader, device: torch.device) -> tuple:
    """
    Evaluate the model on the test dataloader.
    :param model: (callable) the model to evaluate;
    :param test_loader: (torch.utils.data.DataLoader) the test dataloader;
    :param device: (torch.device) the device to use for evaluation;

    :return: res: (tuple) the accuracy (float) on the test dataloader as dim 0,
    the report (dict) of each url with the true and predicted label as dim 1;
    """
    model.eval() # model settato in evaluation (non ho bisogno di calcolare il gradiente)
    correct = 0
    total = 0
    all_predictions = []
    report = []
    with torch.no_grad():
        for (inputs, labels, img_url) in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            for i in range(len(img_url)):
                report.append({
                    'img_url': img_url[i], 
                    'true_label': labels[i].item(), 
                    'predicted_label': predicted[i].item()
                })
            
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    with open('test_report.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['img_url', 'true_label', 'predicted_label'])
        writer.writeheader()
        writer.writerows(report)

    res = (accuracy, report)

    return res