import torch
from PIL import Image
import matplotlib.pyplot as plt
from .. import load_tensor_and_image
from ..params import COCO_INSTANCE_CATEGORY_NAMES


def detect(args, device):
    """Performs image recognition

    Args:
        args (list): The list of arguments provided by the user
        device (object): The pytorch device object

    Returns:
        void
    """
    predictions_per_name = {}
    img_path = args.pretrained[0]
    input_batch, input_image = load_tensor_and_image(img_path, device)
    cnn_model = torch.hub.load(
        'pytorch/vision:v0.4.2',
        'fcn_resnet101',
        pretrained=True).eval()
    with torch.no_grad():
        output = cnn_model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    for j, z in enumerate(output_predictions.byte().cpu().numpy()):
        print('row {}'.format(j))
        print('shape {}'.format(z.shape))
        print('detected {}'.format(z.argmax(0)))
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    r = Image.fromarray(output_predictions.byte().cpu()
                        .numpy()).resize(input_image.size)
    print(input_image.size)
    r.putpalette(colors)
    fig = plt.figure(figsize=(10, 4))
    for i, img in enumerate([r, input_image]):
        fig.add_subplot(1, 2, i + 1)
        plt.imshow(img)
    plt.show()
