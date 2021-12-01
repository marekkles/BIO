import datetime
import os
import time
from matplotlib.pyplot import figure

import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision.transforms import functional as F
from PIL import Image

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Inference', add_help=True)

    parser.add_argument('--model', default='fcn_resnet101', help='model')
    parser.add_argument('--model-path', default='checkpoint.pth', help='path to the model checkpoint')
    parser.add_argument('--device', default='cpu', help='device')
    parser.add_argument('--show', action='store_true', help='show imge plot')
    parser.add_argument('-i', '--input', default='test.jpg', help='file to infere')
    parser.add_argument('-o', '--output', default='inference.png', help='output file')

    args = parser.parse_args()

    model = torchvision.models.segmentation.__dict__[args.model](
        num_classes=3
    )
    model.to(args.device)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)

    image = Image.open(args.input).convert('RGB')
    image = F.resize(image, 100, interpolation=F.InterpolationMode.NEAREST)
    image = F.to_tensor(image)[None,:,:,:]

    with torch.no_grad():
        output = model(image)
        output = torch.nn.functional.softmax(output['out'])[0,:,:,:]

    if args.show:
        from matplotlib.pyplot import imshow, subplot, figure, show
        figure(0)
        ax1 = subplot(212)
        ax1.margins(0.05)           # Default margin is 0.05, value 0 means fit
        ax1.imshow(image[0][0])

        ax3 = subplot(222)
        ax3.margins(2, 2)           # Values >0.0 zoom out
        ax3.imshow(output[1])

        ax2 = subplot(221)
        ax2.margins(2, 2)           # Values >0.0 zoom out
        ax2.imshow(output[0])
        show()

