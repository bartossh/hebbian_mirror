import argparse
import torchvision.models as models


def get_pretrained_nn(nn_type):
    print(nn_type)
    nn_model = models.resnet18(pretrained=True)
    print(nn_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This modules provides \
        hebbian mirror with weights for feeding neuron-network. \
        Weights can be loaded from pre trained neuron-networks \
        from pytorch ecosystem or trained locally')
    parser.add_argument('-p', '--pretrained', type=str, nargs=1,
                        metavar='neron_network_type', default=None,
                        help='Specifies what pre-trained weights \
                        to provide')
    args = parser.parse_args()
    if args.pretrained is not None:
        get_pretrained_nn(args.pretrained[0])


