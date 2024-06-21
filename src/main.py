import sys
sys.path.insert(0,'/home/fv/storage1/qml/DeepLearningBaby/data') ## this is the path of MY PC-change it!

from model import MyVGG
from utils import *
from data.data_loading import *
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

path_2_save_figures = ""  # <- YOUR PATH WHERE 2 SAVE FIGURES!!!
gpu_device = torch.device("cuda")
vgg_net = MyVGG()
print("Train my VGG model...")

train(model=vgg_net,
      train_loader=train_loader,
      val_loader=val_loader,
      epochs=10,
      device=gpu_device,
      learning_rate=0.001,
      seed=8888)


if __name__ == "__main__":
       df, conf = test_classification_report(model=vgg_net, test_loader=test_loader, device=gpu_device)
       print(df)
       Conf_Matrix = ConfusionMatrixDisplay(confusion_matrix=conf).plot()
       plt.savefig(path_2_save_figures+'confusion_matrix.png', dpi=500)
       plt.show()
