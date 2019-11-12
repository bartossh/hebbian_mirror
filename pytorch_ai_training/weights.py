import argparse
import torchvision.models as models
import scipy.stats as stats


vision_models_obj = {
    'resnet18': models.resnet18(pretrained=True),
    'alexnet': models.alexnet(pretrained=True),
    'vgg16': models.vgg16(pretrained=True),
    'squeezenet': models.squeezenet1_0(pretrained=True),
    'densenet': models.densenet161(pretrained=True),
    'inception': models.inception_v3(pretrained=True),
    'googlenet': models.googlenet(pretrained=True),
    'shufflenet': models.shufflenet_v2_x1_0(pretrained=True),
    'mobilenet': models.mobilenet_v2(pretrained=True),
    'resnext50_32x4d': models.resnext50_32x4d(pretrained=True),
    'wide_resnet50_2': models.wide_resnet50_2(pretrained=True),
    'mnasnet': models.mnasnet1_0(pretrained=True)
}


def get_pretrained_nn(nn_type):
    nn_model = vision_models_obj.get(nn_type, 'resnet18')
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
