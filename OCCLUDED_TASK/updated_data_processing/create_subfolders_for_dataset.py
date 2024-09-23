import os, pandas as pd
import shutil


## prova
'''
train_folder = "/Users/francescoaldoventurelli/Desktop/PAZIENTI_CROPPED/train/"
class1, class2 = train_folder + "/class1", train_folder + "/class2"
csv = train_folder + "train_labels.csv"


## nome nel csv = nome dell'immagine
df = pd.DataFrame(pd.read_csv(csv))

for i in range(len(df["label"])):
    if df["label"][i] == 1:
        shutil.copy(train_folder + df["filename"][i], class1)
    else:
        shutil.copy(train_folder + df["filename"][i], class2)'''


path= "/Users/francescoaldoventurelli/Desktop/PAZIENTI_CROPPED"

def movefiles(source_path: str, csv_name: str):
    class1, class2 = source_path + "/class1", source_path + "/class2"
    csv = source_path + csv_name
    df = pd.DataFrame(pd.read_csv(csv))
    for i in range(len(df["label"])):
        if df["label"][i] == 1:
            shutil.copy(source_path + df["filename"][i], class1)
        else:
            shutil.copy(source_path + df["filename"][i], class2)


movefiles(source_path=path+"/test/", csv_name="test_labels.csv") ## for test
movefiles(source_path=path+"/val/", csv_name="val_labels.csv") # validation
