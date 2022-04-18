
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
#drn code imbibe
import drn
# import data_transforms as transforms


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


def create_label_colormap():
    """Creates a label colormap used in Cityscapes segmentation benchmark.

    Returns:
        A Colormap for visualizing segmentation results.
    """
    colormap = np.array([
        [128,  64, 128],
        [244,  35, 232],
        [ 70,  70,  70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170,  30],
        [220, 220,   0],
        [107, 142,  35],
        [152, 251, 152],
        [ 70, 130, 180],
        [220,  20,  60],
        [255,   0,   0],
        [  0,   0, 142],
        [  0,   0,  70],
        [  0,  60, 100],
        [  0,  80, 100],
        [  0,   0, 230],
        [119,  11,  32],
        [  0,   0,   0]], dtype=np.uint8)
    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
        label: A 2D array with integer type, storing the segmentation label.

    Returns:
        result: A 2D array with floating type. The element of the array
            is the color indexed by the corresponding element in the input label
            to the PASCAL color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
            map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]

def vis_segmentation(image, seg_map):
    """Visualizes input image, segmentation map and overlay view."""
    plt.figure(figsize=(20, 4))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')

    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.show()
    
LABEL_NAMES = np.asarray([
'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck',
'bus', 'train', 'motorcycle', 'bicycle', 'void'])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

SAMPLE_IMAGE = 'mit_driveseg_sample.png'
if not os.path.isfile(SAMPLE_IMAGE):
    print('downloading the sample image...')
    SAMPLE_IMAGE = urllib.request.urlretrieve('https://github.com/lexfridman/mit-deep-learning/blob/master/tutorial_driving_scene_segmentation/mit_driveseg_sample.png?raw=true')[0]
print('running deeplab on the sample image...')

def run_visualization(SAMPLE_IMAGE, MODEL):
    """Inferences DeepLab model and visualizes result."""
    original_im = Image.open(SAMPLE_IMAGE)
    seg_map = MODEL.run(original_im)
    vis_segmentation(original_im, seg_map)

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
    parser.add_argument('--pretrained', 
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

    # # torch.manual_seed(args.seed)

    # # device = torch.device("cuda" if use_cuda else "cpu")

    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    if args.inference:

        for k, v in args.__dict__.items():
            print(k, ':', v)

        single_model = DRNSeg(args.arch, args.classes, pretrained_model=None,
                            pretrained=False)
        print("after model init")
        # model = torch.nn.DataParallel(model)
        # checkpoint = torch.load(args.pretrained, map_location="cpu")
        # # start_epoch = checkpoint['epoch']
        # # #best_prec1 = checkpoint['best_prec1']
        # # best_miou = checkpoint['best_miou']
        # model.load_state_dict(checkpoint['state_dict'])
        if args.pretrained:
            single_model.load_state_dict(torch.load(args.pretrained))
        # model = torch.nn.DataParallel(single_model).cuda()
        model = single_model
        video_path = "sample.mp4"
        FrameCapture(video_path, model, vie=True)


if __name__ == '__main__':
    main()
