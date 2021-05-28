"""
Functions for image loading and processing.
"""

import os

import numpy as np
import matplotlib.image as mpimg

import helpers as h


def load_batch_CXR14(p, sample_IDs, data_meta):
    """
    Load a batch of images with indices specified by sample_IDs.

    """

    images = []
    text_labels = []
    GT = []
        
    for idx, sample_index in enumerate(sample_IDs):
        
        image_filename = data_meta.at[sample_index, 'Image Index']

        path = os.path.join(p.data_path, '%s/%s' % (p.images_folder, image_filename))

        image = np.squeeze(mpimg.imread(path)*255)
        image = image.reshape([1] + list(image.shape))

        label_text = data_meta.at[sample_index, 'Finding Labels']
        _, label_vector = h.label2vector_CXR14(label_text)

        images.append(image)
        text_labels.append(label_text)
        GT.append(label_vector)
        
    images = np.array(images)
    GT = np.array(GT, dtype = int)

    # crop out the center of the image to match the network input resolution:
    old_center = np.array(images.shape[-3:]) // 2
    images = extract_patch(images, old_center, [1] + list(p.arch_params.input_res))
    
    images = normalize_CXR14(images)

    return images, GT, text_labels

def normalize_CXR14(images):
    """
    Rescales images between -1 and 1.

    """
    
    images = images / 255. * 2 - 1
    
    return images

def shuffle_array(arr):
    """
    Spacially shuffles a given array.

    """
    
    flat_noise = np.ndarray.flatten(arr)
    
    np.random.shuffle(flat_noise)
    
    permuted = flat_noise.reshape(arr.shape)
    
    return permuted

def extract_patch(image, center, psize):
    """
    Extracts (a zero-padded) patch from a given image given its size and center.

    """
    
    center = np.array(center, dtype=int)
    psize = np.array(psize, dtype=int)
    
    # patch coordinates in the image
    imin = np.maximum(np.zeros(3, dtype = int), center - psize // 2)
    imax = np.maximum(np.minimum(np.array(image.shape[-3:], dtype=int), center + np.ceil(psize / 2.).astype(int)), imin)
    
    # patch coordinates in the patch
    pmin = psize // 2 - (center - imin)
    pmax = pmin + imax - imin
    
    assert(not np.any(pmin < 0))
    assert(not np.any(pmax > np.array(psize, dtype = int)))
    
    patch = np.zeros(tuple(list(image.shape[:-3]) + list(psize)), dtype = image.dtype)

    patch[..., pmin[0]:pmax[0], pmin[1]:pmax[1], pmin[2]:pmax[2]] \
        = image[..., imin[0]:imax[0], imin[1]:imax[1], imin[2]:imax[2]]
        
    return patch