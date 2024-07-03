from vgg_custom_tune_ready import MyVGG
import torch
import torch.nn as nn
import torch.optim as optim
import os
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from data_loader import new_train_dataset
import logging
import tempfile



storage_path = "/home/fv/ray_results"
exp_name = "tune_analyzing_results"
### Inseriamo dei logs per vedere se funziona e monitorare il tutto


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


config = {
        "batch_size": tune.choice([32, 64, 128, 256]),
        "learning_rate": tune.choice([1e-2, 1e-3, 1e-4]),
    }

model = MyVGG()
seed = 8888
epochs = 10
device = torch.device("cuda:0")
#tuning_directory = "/home/fv/storage1/qml/DeepLearningBaby/VGG16-preTrained/tuning_model"


def evaluate_Tuning(model, val_loader) -> None:
    """
    Evaluate both loss and accuracy on a specific set, in this case it should be the
    validation set. The loss and accuracy are computed with torch.no_grad(), i.e.
    in the evaluation mode, without computing the gradient at each epoch.
    :return: (tuple) loss and accuracy of the model after evaluation.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total_instances = 0
    criterion = nn.CrossEntropyLoss()
    for inputs, labels in val_loader:  ### era il contrario prima (sopra with torch.no_grad()) e sotto il for inputs, label
        with torch.no_grad():
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            classifications = torch.argmax(outputs, dim=1)
            correct += (classifications == labels).sum().item()
            total_instances += len(inputs)
    val_loss = total_loss / len(val_loader)
    val_acc = correct / total_instances
    logger.info(f"Evaluation completed - valid_loss: {val_loss:.2f}, val_acc: {val_acc:.2f}")



def train_fn(config) -> None:
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
    model = MyVGG()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
    val_size = 0.3 * len(new_train_dataset)
    train_size = len(new_train_dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(new_train_dataset, [int(train_size), int(val_size)])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size"], shuffle=False)
    # test_loader = torch.utils.data.DataLoader(new_test_dataset, batch_size=config["batch_size"], shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config["batch_size"], shuffle=False)

    for epoch in range(epochs):
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
        model.eval()
        total_loss = 0.0
        correct = 0
        total_instances = 0
        criterion = nn.CrossEntropyLoss()
        for inputs, labels in val_loader:  ### era il contrario prima (sopra with torch.no_grad()) e sotto il for inputs, label
            with torch.no_grad():
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                classifications = torch.argmax(outputs, dim=1)
                correct += (classifications == labels).sum().item()
                total_instances += len(inputs)
        val_loss = total_loss / len(val_loader)
        val_acc = correct / total_instances
        logger.info(f"Evaluation completed - valid_loss: {val_loss:.2f}, val_acc: {val_acc:.2f}")
        #if epoch % 5 == 0:
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
            (model.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            #train.report({"checkpoint": checkpoint})
            train.report(
                {"val_loss": total_loss / len(val_loader), "accuracy": correct / total_instances},
                checkpoint=checkpoint
            )
        logger.info(f"Epoch {epoch + 1} - Checkpoint saved")
        #evaluate_Tuning(model, val_loader)


def Tuning(epochs, gpu_per_trials, n):
    scheduler = ASHAScheduler(
        max_t=epochs,
        grace_period=1,
        reduction_factor=2,
    )
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_fn),
            resources={"cpu": 1, "gpu": gpu_per_trials}
        ),
        param_space=config,
        run_config=train.RunConfig(name=exp_name,
        storage_path=storage_path),
        tune_config=tune.TuneConfig(
            metric="accuracy",
            mode="max",
            scheduler=scheduler,
            num_samples=n,
        )
    )
    results = tuner.fit()

    try:
        best_result = results.get_best_result("accuracy", "max")
        best_config = best_result.config
        best_metrics = best_result.metrics
        logger.info(f"Best trial config: {best_config}")
        logger.info(f"Best trial final validation loss: {best_metrics['val_loss']}")
        logger.info(f"Best trial final validation accuracy: {best_metrics['accuracy']}")
    except Exception as e:
        logger.error("Failed to retrieve best trial results")
        logger.error(e)


if __name__ == "__main__":
    Tuning(epochs, gpu_per_trials=1/4, n=5)

## SUL TERMINALE LANCIARE : CUDA_VISIBLE_DEVICES=0 python _.py

# fare un run di prova con la configurazione più grande e guardo quanto mi occupa sul terminale con nvidia-smi