from __future__ import print_function
import argparse
import torchvision.models as models
import scipy.stats as stats
import torch
from PIL import Image
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vision_models_obj = {
    'resnet18': models.resnet18,
    'alexnet': models.alexnet,
    'vgg16': models.vgg16,
    'squeezenet': models.squeezenet1_0,
    'densenet': models.densenet161,
    'inception': models.inception_v3,
    'googlenet': models.googlenet,
    'shufflenet': models.shufflenet_v2_x1_0,
    'mobilenet': models.mobilenet_v2,
    'resnext50_32x4d': models.resnext50_32x4d,
    'wide_resnet50_2': models.wide_resnet50_2,
    'mnasnet': models.mnasnet1_0
}


def load_image(image_path):
    """Load image to tensor

    Args:
        image_path (str): The path to image

    Returns:
        tensor: Pytorch tensor
    """
    imsize = 512 if torch.cuda.is_available() else 128
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()
    ])
    image = Image.open(image_path)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def train(args):
    pass


def detect(args):
    """Performs image recognition

    Args:
        args (list): The list of arguments provided by the user

    Returns:
        void
    """
    model = args.pretrained[0]
    img_path = args.pretrained[1]
    img_tensor = load_image(img_path)
    cnn = vision_models_obj.get(model, 'resnet18')(
        pretrained=True, progress=True)
    print(img_tensor)
    print(cnn)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This module allows to tests \
        different neuron-network modules')
    parser.add_argument('-p', '--pretrained', type=str, nargs=2,
                        metavar=('neron_network_type', 'image-path'),
                        default=('resnet18', None),
                        help='Specifies what pre-trained nn weights \
                        to provide')
    parser.add_argument('-m', '--model', type=str, nargs=2,
                        metavar=('neron_network_type', 'image-path'),
                        default=('resnet18', None),
                        help='Specifies what model of nn we wont to train \
                        for weights')
    args = parser.parse_args()
    if args.pretrained is not None:
        detect(args)
    if args.model is not None:
        train(args)
