from PIL import Image
import os


source_path = "/Users/francescoaldoventurelli/Desktop/PAZIENTI"
new_source_path = "/Users/francescoaldoventurelli/Desktop/PAZIENTI_CROPPED"

def Cropp(path, name: str):
    """Crop an image based on predefined dimensions and save it."""

    im = Image.open(os.path.join(path, name))
    print(im)
    width, height = im.size
    
    left = 158
    top = height / 14
    right = 525
    bottom = 2.7 * height / 3

    im_new = im.crop((left, top, right, bottom))
    im_new.save(os.path.join(new_source_path, name))


for file_name in os.listdir(source_path):
    Cropp(path=source_path, name=file_name)
print("End.")

'''name = "26_8bit_pz45.png"
im = Image.open(os.path.join(source_path, name))
width, height = im.size
    
left = 158
top = height / 14
right = 525
bottom = 2.7 * height / 3

im_new = im.crop((left, top, right, bottom))
im_new.save(os.path.join(new_source_path, name))'''