### rename all imgs of pz 45 and 41
import os


path41 = "/Users/francescoaldoventurelli/Desktop/pz_41_rot"
path45 = "/Users/francescoaldoventurelli/Desktop/pz_45_rot"

def change_name(file: str, new_char: str) -> str:
    """
    Add a specific string to the name of the file.
    """
    new_name = file[:-4] + str(new_char) + ".png"
    return new_name


for file in os.listdir(path41):
    old_file = os.path.join(path41, file)
    new_file = change_name(old_file, "_pz41")
    file = os.rename(old_file, new_file)
