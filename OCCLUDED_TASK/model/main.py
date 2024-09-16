from torch import device
from torch.optim import Adam
from model import ChildreNet, qbits
from utils import train_and_validate, test, criterion
from custom_loader import train_loader, test_loader, val_loader
from open_json import lr, layers, epochs


print(f"Executing the script with {str(criterion)}...")



model = ChildreNet(layers, qbits)



optimizer = Adam(model.parameters(), lr=lr)  



if __name__ == "__main__":
    train_and_validate(model=model,
                       epochs=epochs,
                       train_loader=train_loader,
                       val_loader=val_loader,
                       device=device("cpu"),
                       optimizer=optimizer
                       )
    
    accuracy_test, report = test(model=model, test_loader=test_loader, device=device("cpu"))
    print(report)