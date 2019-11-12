import argparse
import torchvision.models as models
import scipy.stats as stats


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


def get_pretrained_vision_nn(nn_type):
    nn_model = vision_models_obj.get(nn_type, 'resnet18')(pretrained=True,
        progress=True)
    print(nn_model)


def get_vision_nn(nn_type):
    nn_model = vision_models_obj.get(nn_type, 'resnet18')(pretrained=False,
        progress=True)
    print(nn_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This modules provides \
        hebbian mirror with weights for feeding neuron-network. \
        Weights can be loaded from pre trained neuron-networks \
        from pytorch ecosystem or trained locally')
    parser.add_argument('-p', '--pretrained', type=str, nargs=1,
                        metavar='neron_network_type', default=None,
                        help='Specifies what pre-trained nn weights \
                        to provide')
    parser.add_argument('-m', '--model', type=str, nargs=1,
                        metavar='neron_network_type', default=None,
                        help='Specifies what model of nn we wont to train \
                        for weights')
    args = parser.parse_args()
    if args.pretrained is not None:
        get_pretrained_vision_nn(args.pretrained[0])
    if args.model is not None:
        get_vision_nn(args.model[0])
