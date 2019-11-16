import argparse
import torch
from py_src import download_image, detect, train


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This module allows to tests \
        deeplabv3_resnet101 neuron-network module')
    parser.add_argument('-p', '--pretrained', type=str, nargs=1,
                        metavar=('image-path'),
                        default=(None),
                        help='Specifies what image to test against with \
                        pretrained weights')
    parser.add_argument('-t', '--train', type=str, nargs=1,
                        metavar=('image-path'),
                        default=(None),
                        help='Specifies what image to test against with \
                        after training weights')
    parser.add_argument('-l', '--local', type=str, nargs=1,
                        metavar=('image-path'),
                        default=(None),
                        help='Specifies what locally available and train \
                        model of nn we wont to use')
    parser.add_argument('-d', '--download', type=str, nargs=1,
                        metavar=('url'),
                        default=(None),
                        help='Allows to download image from the given url')
    args = parser.parse_args()
    if args.pretrained is not None and args.pretrained[0] is not None:
        detect(args, device)
    if args.train is not None and args.train[0] is not None:
        train(args, device)
    if args.local is not None and args.local[0] is not None:
        print('Not Implemented')
    if args.download is not None and args.download[0] is not None:
        download_image(args.download)
