import urllib
import torch
from PIL import Image
import torchvision.transforms as transforms


def download_image(args):
    """Downloads image from given url

    Args:
        args (str, str) tuple of url path from which image will be downloaded
                        and image name that will be used to write file on disc

    Returns:
        filename (str): Full name of the file.
    """
    url_path, name = args[0], args[1]
    try:
        urllib.URLopener().retrieve(url_path, name + '.jpg')
    except:
        urllib.request.urlretrieve(url_path, name + '.jpg')
    return name


def load_tensor_and_image(image_path, device):
    """Loads image to tensor

    Args:
        image_path (str): The path to image
        device (object): The pytorch device object

    Returns:
        tuple (tensor, image object): Tuple of Pytorch tensor
    """
    input_image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            ),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch.to(device, torch.float), input_image
