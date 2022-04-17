
import argparse
# from matplotlib import image
import numpy as np
from imageio import imread
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchvision import datasets
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import os
import math
import cv2
import torchvision.transforms as T
import PIL.Image as Image

#drn code imbibe
import drn
import data_transforms as transforms


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
    [0, 0, 0]], dtype=np.uint8)


TRIPLET_PALETTE = np.asarray([
    [0, 0, 0, 255],
    [217, 83, 79, 255],
    [91, 192, 222, 255]], dtype=np.uint8)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DRNSeg(nn.Module):
    def __init__(self, model_name, classes, pretrained_model=None,
                 pretrained=True, use_torch_up=False):
        super(DRNSeg, self).__init__()
        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000)
        pmodel = nn.DataParallel(model)
        if pretrained_model is not None:
            pmodel.load_state_dict(pretrained_model)
        self.base = nn.Sequential(*list(model.children())[:-2])

        self.seg = nn.Conv2d(model.out_dim, classes,
                             kernel_size=1, bias=True)
        self.softmax = nn.LogSoftmax()
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        if use_torch_up:
            self.up = nn.UpsamplingBilinear2d(scale_factor=8)
        else:
            up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                    output_padding=0, groups=classes,
                                    bias=False)
            fill_up_weights(up)
            up.weight.requires_grad = False
            self.up = up

    def forward(self, x):
        x = self.base(x)
        x = self.seg(x)
        y = self.up(x)
        return self.softmax(y), x

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param



def FrameCapture(video_path, model, arch=None):
    
    vidObj = cv2.VideoCapture("sample.mp4")
    count = 0
    success = 1
    images = np.zeros((100, 300, 600, 3), dtype=np.int64)
    i=0
    while success:
        success, image = vidObj.read()
        img = Image.fromarray(image, 'RGB')
        transform = T.Resize((300,600)) # tran
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
    ax1=plt.subplot(111)
    figure, ax = plt.subplots(figsize=(20, 16))

    for i in range(len(images)):
        
        img = torch.unsqueeze(images[i], dim=0)
        final = model(img)[0]
        _, pred = torch.max(final, 1)
        output = pred.cpu().data.numpy()
        print("shape of pred", pred.shape)
        # output = model(img)[0].transpose(
        #     1, 2).detach().numpy().argmax(axis=0)

        # # output = model(images)[0].transpose(1, 2).detach().numpy().argmax(axis=0)
        # print("Shape of output",output.shape)
        
        # print(i)
        inp_img=images[i].transpose(0, 2).detach().int().numpy()
        # view(images[i].transpose(0, 2).detach().int().numpy(), output, i)
        # colors = np.array([[128, 0,0], [0,0,128], [0,128,0], [128,128,128],
        #             [128,64,0], [64,0,128], [0,64,128], [0, 0, 0]
        #             ], dtype=np.int)
        color_image = Image.fromarray(CITYSCAPE_PALETTE[pred.squeeze()])
        # color_image = np.zeros(
        #     (output.shape[0], output.shape[1], 304), dtype=np.int)
        # print("shape of color image", color_image.shape)
        # for j in range(19):
        #     color_image[output == j] = CITYSCAPE_PALETTE[j]

        # color_image[output == 255] = CITYSCAPE_PALETTE[-1]
        # print("color image is generated")
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
    parser.add_argument('--pretrained', 
                        default="cityscapes_pretrained/drn_d_22_cityscapes.pth", 
                        metavar='pretrained', help='path to pretrained weights.')
    parser.add_argument('--inference', default=True, metavar='inference',
                        help='To generate predictions on test set.')
    parser.add_argument('--arch', default="drn_d_22")
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

    # # torch.manual_seed(args.seed)

    # # device = torch.device("cuda" if use_cuda else "cpu")

    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    if args.inference:

        for k, v in args.__dict__.items():
            print(k, ':', v)

        single_model = DRNSeg(args.arch, args.classes, pretrained_model=None,
                            pretrained=True)
        print("after model init")
        if args.pretrained:
            single_model.load_state_dict(torch.load(args.pretrained))
        # model = torch.nn.DataParallel(single_model).cuda()
        model = single_model
        video_path = "Road_1101.mp4"
        FrameCapture(video_path, model, vie=True)
        # for i in range(len(image_paths)):
        #     img = torch.unsqueeze(images[i], dim=0)
        #     output = model(img)[0].transpose(
        #         1, 2).detach().numpy().argmax(axis=0)

            # if args.view:
            #     view(images[i].transpose(0, 2).detach().int().numpy(), output)
            #     pred_path = image_paths[i].replace(
            #         'idd20k_lite', 'preds').replace('leftImg8bit/test/', '').replace('_image.jpg', '_label.png')
            #     os.makedirs(os.path.dirname(
            #         os.path.relpath(pred_path)), exist_ok=True)
            #     img = Image.fromarray(output.astype(np.uint8))
            #     img.save(pred_path)

    else:

        # Code for training on the train split of the dataset.

        image_paths = glob('idd20k_lite/leftImg8bit/train/*/*_image.jpg')
        label_paths = [p.replace('leftImg8bit', 'gtFine').replace(
            '_image.jpg', '_label.png') for p in image_paths]

        images = np.zeros((1403, 227, 320, 3), dtype=np.int)
        labels = np.zeros((1403, 227, 320), dtype=np.int)

        for i in range(1403):
            images[i] = imread(image_paths[i])
            labels[i] = imread(label_paths[i])
            labels[i][labels[i] == 255] = 7

        images = torch.tensor(images, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)

        model = Net().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, images, labels, optimizer, epoch)
            scheduler.step()
            torch.save(model.state_dict(), "seg.pt")


if __name__ == '__main__':
    main()
