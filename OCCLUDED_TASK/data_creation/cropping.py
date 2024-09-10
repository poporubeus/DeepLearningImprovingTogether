from PIL import Image
import os


source_path = "/Users/francescoaldoventurelli/Desktop/pz_45_rot"
new_path = "/Users/francescoaldoventurelli/Desktop/new_children_dataset2"

def Cropp(path: str, old_image_name: str, save_name: str):
    """Crop an image based on predefined dimensions and save it."""

    im = Image.open(os.path.join(path, old_image_name))
    width, height = im.size
    
    left = 158
    top = height / 14
    right = 525
    bottom = 2.7 * height / 3

    im_new = im.crop((left, top, right, bottom))
    im_new.save(os.path.join(new_path, save_name + "newd2.png"))


for file_name in os.listdir(source_path):
    Cropp(path=source_path, old_image_name=file_name, save_name=file_name.rsplit('.', 1)[0])
