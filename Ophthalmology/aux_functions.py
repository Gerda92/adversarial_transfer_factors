"""
Auxiliary functions.

@author: CristinaGGonzalo
"""

import numpy as np

def normalize(img):
    '''
    Normalize image (-1 to 1)
    '''
    return (img/255)*2-1
    
def denormalize(img):
    '''
    Denormalize image (0 to 255) as uint8
    '''
    return ((img+1)/2*255).astype(np.uint8)

def to_float(array):
    return array/255

def to_uint8(array):
    scaled = 255 * array
    clipped = np.clip(scaled, 0, 255)
    return clipped.astype(np.uint8)

def stretch_noise(noise):
    return ((noise - np.min(noise))/(np.max(noise)-np.min(noise)))

def get_adv_noise (img, img_adv):
    '''
    Get adversarial noise for visualization
    '''
    return to_uint8(stretch_noise(to_float(img_adv)-to_float(img)))

def ref_nonref(pred, th):
    '''
    Determine if prediction or label is referable or non-referable DR regarding optimal threshold
    '''
    if pred>th:
        return 'Referable DR'
    else:
        return 'Non-referable DR'