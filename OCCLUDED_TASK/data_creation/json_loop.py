import json
import numpy as np


path = "/Users/francescoaldoventurelli/Downloads/pz_41_rot.json"
path2 = "/Users/francescoaldoventurelli/Downloads/pz_45_rot.json"
with open(path2, 'r') as file:
    data = json.load(file)


PatienceDict = {}
ids = []
label_array = []

def FindMeta(item):
    if "meta" in item:  
        return item["meta"].get("text", [])
    return []


for item in data:
    item_id = item.get("id")
    for annotation in item.get("annotations", []):  
        for result in annotation.get("result", []):  
            meta_values = FindMeta(result)  
            print(f"Item ID: {item_id}, Meta Values: {meta_values}") 
            ids.append(item_id)
            if any(val.strip().lower() == "occluded" for val in meta_values):
                #PatienceDict[item_id] = 1
                label_array.append(1)
            else:
                #PatienceDict[item_id] = 0
                label_array.append(0)


#print(len(label_array))

print(label_array)
'''for (person, label) in zip(ids, label_array):
    PatienceDict[person] = label'''

print(len(label_array))

fixed_nodes = 12

def CheckOcclusion(sublist):
    if 1 in sublist:
        return True
    else:
        return False


empty_bool_labels = []

for i in range(0, len(label_array), fixed_nodes):
    sublist = label_array[i:i + fixed_nodes]
    empty_bool_labels.append(CheckOcclusion(sublist))

print(empty_bool_labels)