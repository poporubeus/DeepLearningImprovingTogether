import json
import pandas as pd


path41 = "/Users/francescoaldoventurelli/Downloads/pz_41_rot.json"
path45 = "/Users/francescoaldoventurelli/Downloads/pz_45_rot.json"
with open(path45, 'r') as file:
    data = json.load(file)


PatienceDict = {}
ids = []
label_array = []

def FindMeta(item):
    if "meta" in item:  
        return item['meta'].get('text', [])
    return []




list_url = [item["data"] for item in data]
    



for item in data:
    item_id = item.get("id")
    
    for annotation in item.get("annotations", []): 
        for result in annotation.get("result", []): 
            meta_values = FindMeta(result)
            ids.append(item_id)
            if any(val.strip().lower() == "occluded" for val in meta_values):
                
                label_array.append(0)

            else:

                label_array.append(1)
        
        



print(label_array)


fixed_nodes = 12

def CheckOcclusion(sublist):
    ## 0 è occluso !!!!
    if 0 in sublist:
        return 0
    else:
        return 1


empty_bool_labels = []

for i in range(0, len(label_array), fixed_nodes):
    sublist = label_array[i:i + fixed_nodes]
    empty_bool_labels.append(CheckOcclusion(sublist))

print(empty_bool_labels)


list_of_patience = []
for item in data:
    item_id = item.get("id")
    list_of_patience.append(item_id)


data = {'ID': list_url,
        'Label': empty_bool_labels[:-1]}
DF = pd.DataFrame(data=data, columns=["ID", "Label"])
DF.to_csv("/Users/francescoaldoventurelli/Desktop/paziente45.csv")
print(DF)

