from __future__ import print_function, division
import os
import argparse
import torch
import torch.nn as nn

import torchvision.utils as tv
import torchvision.transforms as transforms
from scipy.misc import imsave

from torch.utils.data import Dataset, DataLoader
from model.cnn_geometric_model import CNNGeometric
from data.pf_dataset import PFDataset
from data.download_datasets import download_PF_willow
from image.normalization import NormalizeImageDict, normalize_image
from util.torch_util import BatchTensorToVars, str_to_bool
from geotnf.transformation import GeometricTnf
from geotnf.point_tnf import *
import matplotlib.pyplot as plt
from skimage import io
from collections import OrderedDict


# for compatibility with Python 2
try:
    input = raw_input
except NameError:
    pass

print('CNNGeometric aerial image demo script')

# Argument parsing
parser = argparse.ArgumentParser(description='CNNGeometric PyTorch implementation')
# Paths
parser.add_argument('--model-aff', type=str, default='trained_models/Static/ai/best_checkpoint_adam_affine_grid_lossdensenet161.pth.tar', help='Trained affine model filename')
parser.add_argument('--feature-extraction-cnn', type=str, default='densenet161', help='Feature extraction architecture: vgg/resnet101')
parser.add_argument('--pf-path', type=str, default='datasets/all', help='Path to PF dataset')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()

do_aff = not args.model_aff==''

# Create model
print('Creating CNN model...')
if do_aff:
    model_aff = CNNGeometric(use_cuda=use_cuda,geometric_model='affine',feature_extraction_cnn=args.feature_extraction_cnn)

# Load trained weights
print('Loading trained model weights...')
if do_aff:
    checkpoint = torch.load(args.model_aff, map_location=lambda storage, loc: storage)
    checkpoint['state_dict'] = OrderedDict([(k.replace('densenet161', 'model'), v) for k, v in checkpoint['state_dict'].items()])
    model_aff.load_state_dict(checkpoint['state_dict'])

# Dataset and dataloader
dataset = PFDataset(csv_file=os.path.join(args.pf_path, 'test_pairs_all.csv'),
                    training_image_path=args.pf_path,
                    transform=NormalizeImageDict(['source_image','target_image']))
dataloader = DataLoader(dataset, batch_size=1,
                        shuffle=False, num_workers=4)
batchTensorToVars = BatchTensorToVars(use_cuda=use_cuda)

# Instantiate point transformer
pt = PointTnf(use_cuda=use_cuda)

# Instatiate image transformers
affTnf = GeometricTnf(geometric_model='affine', use_cuda=use_cuda)



for i, batch in enumerate(dataloader):
    # get random batch of size 1
    batch = batchTensorToVars(batch)
    
    source_im_size = batch['source_im_size']
    target_im_size = batch['target_im_size']

    source_points = batch['source_points']
    target_points = batch['target_points']
    
    # warp points with estimated transformations
    target_points_norm = PointsToUnitCoords(target_points,target_im_size)
    
    if do_aff:
        model_aff.eval()
        
    # Evaluate models
    if do_aff:
        theta_aff=model_aff(batch)
        warped_image_aff = affTnf(batch['source_image'],theta_aff.view(-1,2,3))
        print(theta_aff)
        print(theta_aff.view(-1,2,3))


    # Un-normalize images and convert to numpy
    source_image = normalize_image(batch['source_image'],forward=False)
    source_image = source_image.data.squeeze(0).transpose(0,1).transpose(1,2).cpu().numpy()
    target_image = normalize_image(batch['target_image'],forward=False)
    target_image = target_image.data.squeeze(0).transpose(0,1).transpose(1,2).cpu().numpy()
    
    if do_aff:
        warped_image_aff = normalize_image(warped_image_aff,forward=False)
        warped_image_aff = warped_image_aff.data.squeeze(0).transpose(0,1).transpose(1,2).cpu().numpy()

    # check if display is available
    exit_val = os.system('python -c "import matplotlib.pyplot as plt;plt.figure()"  > /dev/null 2>&1')
    display_avail = exit_val==0

    if display_avail:
        N_subplots = 2+int(do_aff)
        fig, axs = plt.subplots(1,N_subplots)
        axs[0].imshow(source_image)
        axs[0].set_title('src')
        axs[1].imshow(target_image)
        axs[1].set_title('tgt')
        subplot_idx = 2
        if do_aff:
            axs[subplot_idx].imshow(warped_image_aff)
            axs[subplot_idx].set_title('aff')
            imsave('source.jpg', source_image)
            imsave('target.jpg', target_image)
            imsave('result.jpg', warped_image_aff)
            subplot_idx +=1

        for i in range(N_subplots):
            axs[i].axis('off')
        print('Showing results. Close figure window to continue')
        plt.show()
    else:
        print('No display found. Writing results to:')
        fn_src = 'source.png'
        print(fn_src)
        io.imsave(fn_src, source_image)
        fn_tgt = 'target.png'
        print(fn_tgt)
        io.imsave(fn_tgt, target_image)
        if do_aff:
            fn_aff = 'result_aff.png'
            print(fn_aff)
            io.imsave(fn_aff, warped_image_aff)

    res = input('Run for another example ([y]/n): ')
    if res=='n':
        break