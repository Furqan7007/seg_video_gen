
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
# from torchvision import datasets
from torch.optim.lr_scheduler import StepLR
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import os
import math
import cv2
import torchvision.transforms as T
import PIL.Image as Image
import time
# Tensorflow
# !pip install tensorflow==1.13.1
import tensorflow as tf
print(tf.__version__)
# import tensorflow as tf
# print(tf.__version__)

# I/O libraries
import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

# Helper libraries
import matplotlib
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import cv2 as cv
from tqdm import tqdm
import IPython
from sklearn.metrics import confusion_matrix
from tabulate import tabulate

# Comment this out if you want to see Deprecation warnings
import warnings
warnings.simplefilter("ignore", DeprecationWarning)


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        graph_def = None

        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break
        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
        self.sess = tf.Session(graph=self.graph)

    def run(self, image, INPUT_TENSOR_NAME = 'ImageTensor:0', OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'):
        """Runs inference on a single image.

        Args:
            image: A PIL.Image object, raw input image.
            INPUT_TENSOR_NAME: The name of input tensor, default to ImageTensor.
            OUTPUT_TENSOR_NAME: The name of output tensor, default to SemanticPredictions.

        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        target_size = (2049,1025)  # size of Cityscapes images
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            OUTPUT_TENSOR_NAME,
            feed_dict={INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]  # expected batch size = 1
        if len(seg_map.shape) == 2:
            seg_map = np.expand_dims(seg_map,-1)  # need an extra dimension for cv.resize
        seg_map = cv.resize(seg_map, (width,height), interpolation=cv.INTER_NEAREST)
        return seg_map

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

def FrameCapture(video_path, model, vie=None):
    
    vidObj = cv2.VideoCapture(video_path)
    plt.ion()
    figure, ax1 = plt.subplots(figsize=(20, 16))
    images = []
    
    for i in range(30):
         _, frame = vidObj.read()
         img = Image.fromarray(frame[..., ::-1])
        #  print("input image size is ", img.shape)
        #  t = T.Resize(300,600)
        #  img = t( /img)
         images.append(img)
    print(len(images))
    for j in range(len(images)):
        print("at the start")
        start_time = time.time()
        seg_map = model.run(images[j])
        end_time = time.time()
        print("time for generating output", end_time - start_time)
        color_image = label_to_color_image(seg_map).astype(np.uint8)
        print("color image is generated")

        im1=ax1.imshow(images[j])
        im1.set_data(images[j])
        figure.canvas.draw()
        im2=ax1.imshow(color_image,alpha=0.5)
        im2.set_data(color_image)

    # # ax1.imshow(inp_img)
    # # ax1.imshow(color_image)
        figure.canvas.draw()
        figure.canvas.flush_events()
        # plt.imshow(images[j])
        # plt.imshow(color_image, alpha=0.7)
        # plt.axis('off')
        # plt.title('segmentation overlay')
        # # # plt.grid('off')
        # # # plt.tight_layout()
        # # # plt.show()
        print("after plt show")

            
        #  cv2.imwrite(f"frames_output/frame_{count}.jpg", image)
        # count += 1

# SAMPLE_IMAGE = 'mit_driveseg_sample.png'
# if not os.path.isfile(SAMPLE_IMAGE):
#     print('downloading the sample image...')
#     SAMPLE_IMAGE = urllib.request.urlretrieve('https://github.com/lexfridman/mit-deep-learning/blob/master/tutorial_driving_scene_segmentation/mit_driveseg_sample.png?raw=true')[0]
# print('running deeplab on the sample image...')

# def run_visualization(SAMPLE_IMAGE, MODEL):
#     """Inferences DeepLab model and visualizes result."""
#     original_im = Image.open(SAMPLE_IMAGE)
#     seg_map = MODEL.run(original_im)
#     vis_segmentation(original_im, seg_map)


def vis_segmentation_stream(image, seg_map, index):
    """Visualizes segmentation overlay view and stream it with IPython display."""
    plt.figure(figsize=(12, 7))

    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay | frame #%d'%index)
    plt.grid('off')
    plt.tight_layout()

    # Show visualization in a streaming fashion.
    # f = BytesIO()
    # plt.savefig(f, format='jpeg')
    # IPython.display.display(IPython.display.Image(data=f.getvalue()))
    # f.close()
    # plt.close()


def run_visualization_video(frame, index, MODEL):
    """Inferences DeepLab model on a video file and stream the visualization."""
    original_im = Image.fromarray(frame[..., ::-1])
    seg_map = MODEL.run(original_im)
    vis_segmentation_stream(original_im, seg_map, index)


SAMPLE_VIDEO = 'mit_driveseg_sample.mp4'
if not os.path.isfile(SAMPLE_VIDEO): 
    print('downloading the sample video...')
    SAMPLE_VIDEO = urllib.request.urlretrieve('https://github.com/lexfridman/mit-deep-learning/raw/master/tutorial_driving_scene_segmentation/mit_driveseg_sample.mp4')[0]
print('running deeplab on the sample video...')


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Segmentation demo with video')
    parser.add_argument('--pretrained', 
                        default="sem_baseline_exp/cityscapes_drn_d_38_150/checkpoint.pth.tar", 
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

        MODEL_NAME = 'mobilenetv2_coco_cityscapes_trainfine'
        #MODEL_NAME = 'xception65_cityscapes_trainfine'

        _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
        _MODEL_URLS = {
            'mobilenetv2_coco_cityscapes_trainfine':
                'deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz',
            'xception65_cityscapes_trainfine':
                'deeplabv3_cityscapes_train_2018_02_06.tar.gz',
        }
        _TARBALL_NAME = 'deeplab_model.tar.gz'

        model_dir = tempfile.mkdtemp()
        tf.gfile.MakeDirs(model_dir)

        download_path = os.path.join(model_dir, _TARBALL_NAME)
        print('downloading model, this might take a while...')
        urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME], download_path)
        print('download completed! loading DeepLab model...')

        model = DeepLabModel(download_path)
        print('model loaded successfully!')
        FrameCapture(SAMPLE_VIDEO, model, vie=True)


if __name__ == '__main__':
    main()
