using Images, FileIO, ImageView

img_path = "/Users/francescoaldoventurelli/Desktop/pz_41_rot/2901_8bit.png"
img = load(img_path)


# left = 158
# top = height / 14
# right = 525
# bottom = 2.7 * height / 3

function cropping(image_file)
    img_size = size(image_file)
    img_cropped = @view image_file[ :,floor(Int, 1/8*img_size[2]) : floor(Int, 7/8*img_size[2])]
    return img_cropped
end


cropped_fig = cropping(img)
imshow(cropped_fig)