"""
Preprocessing of color fundus images.

@author: CristinaGGonzalo
"""

from PIL import Image
from skimage.measure import label, regionprops
from scipy.ndimage import zoom
import os
import numpy as np
from skimage.filters import threshold_otsu

def get_patch(array, y0, y1, x0, x1):
    '''get patch from array
    y0, y1: min and max y
    x0, x1: min and max x
    
    if y0 or x0 are negative, or if y1 or x1 are larger than the width/height of the array, it will zero pad the result
    size of the result is therefore always (y1-y0, x1-x0)
    '''
    h, w = array.shape[:2]
    rest = array.shape[2:]
    h_out, w_out = y1 - y0, x1 - x0
    result = np.zeros((h_out, w_out) + rest)

    ry0, rx0 = 0, 0
    py0, px0 = y0, x0
    if y0 < 0:
        py0 = 0
        ry0 = -y0 
    if x0 < 0:
        px0 = 0
        rx0 = -x0 
    patch = array[py0:y1, px0:x1]
    ph, pw = patch.shape[:2]
    result[ry0:ry0+ph, rx0:rx0+pw] = patch
    return result, {'ry': (ry0, ry0 + ph), 'rx': (rx0, rx0 + pw), 'py': (py0, y1), 'px': (px0, x1)}

class Preprocess:
    
    def __init__(self, fn, out_size=(512, 512), threshold=5, extended_bbox=False):
        '''
        fn: filename to process
        out_size: tuple (height, width) of the output image
        threshold: values below this value will be used to identify the mask of the background field of view
                    you can use 'otsu' to use an otsu threshold
                    you can use 'min' to derive it from the histogram (it will take the minimum in the histogram as threshold, more or less)
        extended_bbox: whether to crop the image around the detected bounding box and discard pixel information outside this box (False) 
                    or to use all pixels from the input that can be mapped to pixels in the output image (True)
        
        '''
        self.fn = fn
        self.out_size = out_size
        self.h_out, self.w_out = out_size
        self.th = threshold
        self.extended_bbox = extended_bbox
    
    def set_actual_threshold(self):
        if self.th == 'otsu':
            self.threshold = threshold_otsu(self.data_orig[:, :, 1])
        elif self.th == 'min':
            nn, ii = np.histogram(self.data_orig[:,:,1].flatten(), bins=range(0, 255, 5))
            nn[nn < np.median(nn)] = 0 # ignore small peaks
            maxima = argrelextrema(nn, np.greater, mode='wrap')[0]
            minimum = ii[nn[maxima[0]:maxima[1]].argmin()]
            self.threshold = minimum + 5
        else:
            # check what the background color is
            background_color_green = np.min(self.data_orig[:, :, 1])

            if background_color_green >= self.th:
                self.threshold = background_color_green + self.th
            else:
                self.threshold = self.th

    def get_bbox(self):
        h, w = self.data_orig.shape[:2]
        
        self.mask_orig = self.data_orig[:, :, 1] > self.threshold

        labels = label(self.mask_orig)
        # get properties (area and bbox) per connected component
        props = regionprops(labels)
        # find largest connected component
        largest_area_index = np.argmax([p.area for p in props])
        # pixels belonging to the bounding box are in the half-open interval [min_row; max_row) and [min_col; max_col)
        self.min_row, self.min_col, self.max_row, self.max_col = props[largest_area_index].bbox 
        self.data_bbox = self.data_orig[self.min_row:self.max_row, self.min_col:self.max_col]

        bb_h = self.max_row - self.min_row
        bb_w = self.max_col - self.min_col
        if bb_h > bb_w:
            d = (bb_h - bb_w)//2
            d1 = bb_h - bb_w - d
            y0, y1 = self.min_row, self.max_row
            x0, x1 = self.min_col - d, self.max_col + d1
        else:
            d = (bb_w - bb_h)//2
            d1 = bb_w - bb_h - d
            y0, y1 = self.min_row - d, self.max_row + d1
            x0, x1 = self.min_col, self.max_col
            
        self.data_bbox2, self.patch_params = get_patch(self.data_orig, y0, y1, x0, x1)

    def zoom(self):
        in_shape = self.data_bbox.shape

        h_in, w_in = in_shape[:2]
        # self.radius is half the width (or height) of the image
        self.radius = max(self.h_out, self.w_out) / 2

        # scale needed to get the max of (w, h) to fit to the out_size
        self.s = min(self.h_out / h_in, self.w_out / w_in)

        # order 0 will do nearest neighbor interpolation
        self.data_zoom = zoom(self.data_bbox[:,:,:3], (self.s, self.s, 1), order=0) 
        self.data_zoom2 = zoom(self.data_bbox2[:,:,:3], (self.s, self.s, 1), order=0) 

    def scale(self):
        in_shape = self.data_bbox.shape
        h, w = self.data_zoom.shape[:2]
        self.start_h = (self.h_out - h) // 2 
        self.start_w = (self.w_out - w) // 2

        if self.extended_bbox:
            self.data_scale = self.data_zoom2
            self.data_scale = self.data_scale.astype(self.data_orig.dtype)
        else:
            self.data_scale = np.zeros(self.out_size + (3,))
            self.data_scale[self.start_h:self.start_h + h, self.start_w:self.start_w + w] = self.data_zoom
            
        init_data_scale_mask = self.data_scale[:,:,1] > self.threshold
           

    def process(self):
        self.data_orig = np.array(Image.open(self.fn))
        self.set_actual_threshold()
        self.get_bbox()
        self.zoom()
        self.scale()

    
    def get_filename(self, folder, filename=None, ext='.png'):
        if filename is None:
            basename = os.path.basename(os.path.splitext(self.fn)[0])
        else:
            basename = filename
        return os.path.join(folder, basename + ext)
    
    def export_rgb(self, folder, filename=None, ext='.png'):
        out_filename = self.get_filename(folder, filename, ext)
        Image.fromarray(self.data_scale.astype(np.uint8)).save(out_filename)
        
        
        