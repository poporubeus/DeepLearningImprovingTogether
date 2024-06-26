from PIL import Image


def UpSample(image, size: int) -> Image:
    """
    Upsample the image with given size you choose and returns the resized image.
    :param image: torch.dataset.data - the image to be upsampled;
    :param size: int - the desired size;
    :return: PIL.Image - the resized image.
    """
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((size, size), Image.BILINEAR)
    return resized_image