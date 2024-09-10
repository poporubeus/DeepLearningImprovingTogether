import pandas as pd
import os



sorgent_path = "/Users/francescoaldoventurelli/Desktop/children_dataset/"
idx, labels = [], []
csv_labels_file = {}

for file in os.listdir(sorgent_path):
    idx.append(file)
    labels.append(1 if "d2.png" in file else 0)


csv_labels_file = {idx[i]: labels[i] for i in range(len(idx))}
csv_labels_df = pd.DataFrame({
    'id': idx,
    'label': labels
})

csv_labels_df.to_csv(os.path.join(sorgent_path, "children_data.csv"), index=False)