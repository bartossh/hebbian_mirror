from __future__ import print_function
import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import urllib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

avaliable_vision_models = [
    'fcn_resnet50', 'fcn_resnet101',
    'deeplabv3_resnet50', 'deeplabv3_resnet101'
    ]


def show_models():
    """Prints available neuron networks to console
    """
    print('\n Available models names: \n')
    for model in avaliable_vision_models:
        print('- {}'.format(model))
    print('\n' + ' - ' * 10 + '\n')


def download_image(args):
    """Downloads image from given url

    Args:
        args (str, str) tuple of url path from which image will be downloaded
                        and image name that will be used to write file on disc

    Returns:
        filename (str): Full name of the file.
    """
    url_path, name = args[0], args[1]
    try: urllib.URLopener().retrieve(url_path, name + '.jpg')
    except: urllib.request.urlretrieve(url_path, name + '.jpg')
    return name


def load_tensor_and_image(image_path):
    """Loads image to tensor

    Args:
        image_path (str): The path to image

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


def train(args):
    pass


def detect(args):
    """Performs image recognition

    Args:
        args (list): The list of arguments provided by the user

    Returns:
        void
    """
    model_name = args.pretrained[0]
    img_path = args.pretrained[1]
    input_batch, input_image = load_tensor_and_image(img_path)
    if model_name not in avaliable_vision_models:
        raise Exception('Model not available')
    cnn_model = torch.hub.load(
        'pytorch/vision:v0.4.2',
        model_name,
        pretrained=True).eval()
    with torch.no_grad():
        output = cnn_model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    r = Image.fromarray(output_predictions.byte().cpu()
        .numpy()).resize(input_image.size)
    r.putpalette(colors)
    plt.imshow(r)
    plt.show()


if __name__ == '__main__':
    show_models()
    parser = argparse.ArgumentParser(description='This module allows to tests \
        different neuron-network modules')
    parser.add_argument('-p', '--pretrained', type=str, nargs=2,
                        metavar=('neron_network_type', 'image-path'),
                        default=('fcn_resnet101', None),
                        help='Specifies what pre-trained nn with weights \
                        to provide')
    parser.add_argument('-t', '--train', type=str, nargs=2,
                        metavar=('neron_network_type', 'image-path'),
                        default=('fcn_resnet101', None),
                        help='Specifies what model of nn we wont to train \
                        for weights')
    parser.add_argument('-l', '--local', type=str, nargs=2,
                        metavar=('neron_network_type', 'image-path'),
                        default=('fcn_resnet101', None),
                        help='Specifies what locally available and train model \
                        model of nn we wont to use')
    parser.add_argument('-d', '--download', type=str, nargs=2,
                        metavar=('neron_network_type', 'image-path'),
                        default=('fcn_resnet101', None),
                        help='Specifies what locally available and train model \
                        model of nn we wont to use')
    args = parser.parse_args()
    if args.pretrained is not None and args.pretrained[1] is not None:
        detect(args)
    if args.train is not None and args.train[1] is not None:
        train(args)
    if args.local is not None and args.local[1] is not None:
        print('Not Implemented')
    if args.download is not None and args.download[1] is not None:
        download_image(args.download)
