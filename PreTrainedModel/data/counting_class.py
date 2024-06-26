from data_loading import train_loader, val_loader, test_loader
import torch


def get_binary_distribution(dataset: torch.utils.data.dataset) -> dict:
    """Count the number of items per class in a dataset."""

    class0 = []
    class1 = []
    for batch_idx, (data, targets) in enumerate(dataset):
        class0.append(list(targets.numpy()).count(0))
        class1.append(list(targets.numpy()).count(1))
    return {"Label 0": sum(class0), "Label 1": sum(class1)}


if __name__ == "__main__":
    print("Train:", get_binary_distribution(train_loader))
    print("Test:", get_binary_distribution(test_loader))
    print("Val:", get_binary_distribution(val_loader))