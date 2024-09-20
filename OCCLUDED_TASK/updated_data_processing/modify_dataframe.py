import pandas as pd


path41_csv = "/Users/francescoaldoventurelli/Desktop/paziente41.csv"
path45_csv = "/Users/francescoaldoventurelli/Desktop/paziente45.csv"
index_to_cut = 34  ### questa è ua stringa di carattere di lunghezza sempre uguale (34 char) da togliere,
# proveniente dall'url nel json
final_index = -2 # questa invece corrisponde ad una ' e } da eliminare

#df41 = pd.DataFrame(pd.read_csv(path41_csv))
df45 = pd.DataFrame(pd.read_csv(path45_csv))

def change_name(file: str, new_char: str) -> str:
    """
    Add a specific string to the name of the file
    """
    new_name = file[:-4] + str(new_char) + ".png"
    return new_name



def cut_name(df: pd.DataFrame, n_item: int) -> list:
    """
    Cut the 34-th part of the unnecessary url from the file which will be looped through.
    """
    return list(df["ID"][n_item][index_to_cut:])



def convert_to_string(s: str) -> str: 
    """
    Transform a list of chars into a single string.
    """
    str1 = "" 
    return(str1.join(s)) 

#print(list(df41["ID"][10][34:]))

new_id_list = []
for i in range(len(df45)):
    cutted_name = cut_name(df45, i)
    cutted_name_2 = convert_to_string(cutted_name[:-2])
    new_id_list.append(change_name(cutted_name_2, "_pz45"))
    
    #new_id_list.append(convert_to_string(cut_final(df41, i)))


new_data45 = {"ID": new_id_list,
              "Label": df45["Label"]}

new_df45 = pd.DataFrame(new_data45)
new_df45.to_csv("/Users/francescoaldoventurelli/Desktop/paziente45_new.csv")