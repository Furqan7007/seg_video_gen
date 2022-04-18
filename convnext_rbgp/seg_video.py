
import argparse
# from matplotlib import image
import matplotlib
import numpy as np
from imageio import imread
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import os
import math
import cv2
import torchvision.transforms as T
import PIL.Image as Image
import json


#convnext code imbibe
import mmcv
# import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

# from backbone import beit
from backbone import convnext


CITYSCAPE_PALETTE = np.asarray([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]], dtype=np.int)


TRIPLET_PALETTE = np.asarray([
    [0, 0, 0, 255],
    [217, 83, 79, 255],
    [91, 192, 222, 255]], dtype=np.uint8)


def FrameCapture(video_path, model, vie=None):
    
    vidObj = cv2.VideoCapture(video_path)
    count = 0
    success = 1
    images = np.zeros((100, 300, 600, 3), dtype=np.int64)
    i=0
    while success:
        success, image = vidObj.read()
        img = Image.fromarray(image, 'RGB')
        info = json.load(open('info.json'))
        normalize = transforms.Normalize(mean=info['mean'], std=info['std'])
        scales = [0.5, 0.75, 1.25, 1.5, 1.75]
        transform = T.Resize((300,600)) # tran
        # transform = transforms.Compose([
        #     transforms.Resize((600,300)),
        #     transforms.ToTensor(),
        #     normalize])
        resized_img = transform(img)
        images[i] = resized_img
        i += 1
        if i == 100:
            break
        
    print(f"Shape of each frame is {images.shape}")
    images = torch.tensor(images, dtype=torch.float)
    
    images = images.transpose(1, 3)
    print(len(images))
    # print(success, image)
    print(f"shape of images after transpose is {images[1].shape}")

    plt.ion()
    # ax1=plt.subplot(111)
    figure, ax1 = plt.subplots(figsize=(20, 16))
    model.eval()
    for i in range(len(images)):
        img = torch.unsqueeze(images[i], dim=0)
        # img = images[i]
        final = model(img)[0].transpose(1,3)
        final = torch.squeeze(final, dim=0)
        print("shape of final",final.shape)
        _, pred = torch.max(final, -1)
        output = pred.cpu().data.numpy()
        print("shape of pred", output.shape)
        # output = model(img)[0].transpose(
        #     1, 2).detach().numpy().argmax(axis=0)

        # # output = model(images)[0].transpose(1, 2).detach().numpy().argmax(axis=0)
        # print("Shape of output",output.shape)
        
        # print(i)
        inp_img=images[i].transpose(0, 2).detach().int().numpy()
        print("shape of input image", inp_img.shape)
        # view(images[i].transpose(0, 2).detach().int().numpy(), output, i)
        # colors = np.array([[128, 0,0], [0,0,128], [0,128,0], [128,128,128],
        #             [128,64,0], [64,0,128], [0,64,128], [0, 0, 0]
        #             ], dtype=np.int)
        # color_image = Image.fromarray(CITYSCAPE_PALETTE[pred.squeeze()])
        color_image = np.zeros(
            (output.shape[0], output.shape[1], 3), dtype=np.int)
        print("shape of color image", color_image.shape)
        for j in range(19):
            # print(CITYSCAPE_PALETTE[j])
            # print(color_image[output])
            color_image[output == j] = CITYSCAPE_PALETTE[j]

        color_image[output == 255] = CITYSCAPE_PALETTE[-1]
        print("color image is generated")
        # print('shapes ', image_frame.shape, color_image.shape)
        # print(type(images[i]))
        # print(images[i])
        from skimage import color

        im1=ax1.imshow(inp_img)
        im1.set_data(inp_img)
        figure.canvas.draw()
        im2=ax1.imshow(color_image,alpha=0.5)
        im2.set_data(color_image)

        # ax1.imshow(inp_img)
        # ax1.imshow(color_image)
        figure.canvas.draw()
        figure.canvas.flush_events()

        
    cv2.imwrite(f"frames_output/frame_{count}.jpg", image)
    # count += 1



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Segmentation demo with video')
    parser.add_argument('--checkpoint', 
                        default="sem_baseline_exp/cityscapes_drn_d_38_150/model_best.pth.tar", 
                        metavar='pretrained', help='path to pretrained weights.')
    parser.add_argument('--inference', default=True, metavar='inference',
                        help='To generate predictions on test set.')
    parser.add_argument('--arch', default="drn_d_38")
    parser.add_argument('-d', '--video_path', default=None, required=False)
    parser.add_argument('-c', '--classes', default=19, type=int)
    parser.add_argument('-s', '--crop-size', default=0, type=int)
    parser.add_argument('--view', default=True, metavar='inference',
                        help='View predictions at inference.')
    parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                        help='input batch size for training (default: 200)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--bn-sync', action='store_true')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # # torch.manual_seed(args.seed)

    # # device = torch.device("cuda" if use_cuda else "cpu")

    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    if args.inference:

        for k, v in args.__dict__.items():
            print(k, ':', v)
        
        cfg.model.train_cfg = None
        model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = checkpoint['meta']['PALETTE']

        # single_model = DRNSeg(args.arch, args.classes, pretrained_model=None,
        #                     pretrained=False)
        print("after model init")
        # model = torch.nn.DataParallel(model)
        # checkpoint = torch.load(args.pretrained, map_location="cpu")
        # # start_epoch = checkpoint['epoch']
        # # #best_prec1 = checkpoint['best_prec1']
        # # best_miou = checkpoint['best_miou']
        # model.load_state_dict(checkpoint['state_dict'])
        # if args.pretrained:
        #     single_model.load_state_dict(torch.load(args.pretrained))
        # model = torch.nn.DataParallel(single_model).cuda()
        # model = single_model
        video_path = "sample.mp4"
        FrameCapture(video_path, model, vie=True)


if __name__ == '__main__':
    main()
