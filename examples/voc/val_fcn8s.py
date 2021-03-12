#!/usr/bin/env python

import argparse
import datetime
import os
import os.path as osp

import torch
import yaml

import sys
sys.path.append("./torchfcn/models/")
sys.path.append("./torchfcn/datasets/")
from fcn8s import FCN8s
from DeepFashion import DeepFashionDataset
import skimage.io
import PIL.Image as Image
import cv2
import numpy as np


here = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-g', '--gpu', default='0', type=int, required=True, help='gpu id')
    parser.add_argument('--resume', default=None, help='checkpoint path')
    # configurations (same configuration as original work)
    # https://github.com/shelhamer/fcn.berkeleyvision.org

    parser.add_argument(
        '--pretrained-model',
        default=None,
        help='pretrained model of FCN16s',
    )
    args = parser.parse_args()

    args.model = 'FCN8s'

    now = datetime.datetime.now()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset

    root = osp.expanduser('~/data/datasets/dp')
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

    val_loader = torch.utils.data.DataLoader(
        DeepFashionDataset(root, split='val', transform=True),
        batch_size=8, shuffle=False, **kwargs)

    # 2. model

    model = FCN8s(n_class=20)
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    if cuda:
        model = model.cuda()


    print("validate()")
    model.eval()

    n_class = 20

    visualizations = []
    label_trues, label_preds = [], []
    i = 0
    for batch_idx, (data, target) in enumerate(val_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()

        with torch.no_grad():
            print("model")
            score = model(data)

        imgs = data.data.cpu()
        # print(score.shape)  # torch.Size([8, 20, 512, 320])
        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = target.data.cpu()
        # lbl_pred: (8, 512, 320)
        # lbl_true: torch.Size([8, 512, 320])
#         print("lbl_pred: %s" % (str(lbl_pred.shape)))
#         print("lbl_true: %s" % str(lbl_true.shape))

        for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
            img, lt = val_loader.dataset.untransform(img, lt)
            # img: (512, 320, 3)
            # lt: (512, 320)
#             print(lp[lp != 0])
#             print("lp.shape :%s" % str(lp.shape))
            lp = np.array(lp * 255, dtype=np.uint8)
            # cv2.imwrite('1.jpg', Image.fromarray(lp))
            i += 1


if __name__ == '__main__':
    main()

