import argparse
import logging
import os

import scipy.io
import sys
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import numpy as np

from marrnet import MarrNet, summary


class WriteLog(object):
    def __init__(self, logger, log_level):
        super(WriteLog, self).__init__()
        self.logger = logger
        self.log_level = log_level

    def write(self, s):
        if s.endswith('\n'):
            s = s[:-1]
        if len(s) == 0:
            return
        self.logger.log(msg=s, level=self.log_level)


def load_image(path):
    return Image.open(path)


def preprocess(normalize_mean, normalize_std, scale_dim=256):
    preprocess = transforms.Compose([
        transforms.Scale(scale_dim),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])
    return preprocess


def main():
    parser = argparse.ArgumentParser('MarrNet')
    parser.add_argument('--imgname', type=str, default='image/chair_1.png')
    parser.add_argument('--imgdim', type=int, default=256)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    args.normalize_mean = [0.485, 0.456, 0.406]
    args.normalize_std = [0.229, 0.224, 0.225]
    log_level = logging.getLevelName(args.log_level)
    logging.basicConfig(stream=sys.stdout, level=log_level)
    logger = logging.getLogger()
    logger.info('Running with args %s', vars(args))

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    # input image
    img = load_image(args.imgname)
    img = preprocess(args.normalize_mean, args.normalize_std, scale_dim=args.imgdim)(img)
    img = img.unsqueeze(0)
    img = Variable(img)

    # run model
    model = MarrNet()
    model.eval()
    if log_level <= logging.DEBUG:
        summary(model, img, file=WriteLog(logger, logging.DEBUG))
    output = model(img)
    output = output.squeeze(1)
    output = output.data.numpy().astype(np.double)

    # store output
    savepath = os.path.join(args.output_dir, os.path.splitext(args.imgname)[0] + '.mat')
    scipy.io.savemat(savepath, {'voxels': output, 'args': args})
    logger.info('Saved to %s', savepath)


if __name__ == '__main__':
    main()
