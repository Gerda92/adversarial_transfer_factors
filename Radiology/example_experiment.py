
from omegaconf import OmegaConf

import numpy as np

from compute_attacks import test_adversarial_attack
from helpers import plot_batch


#%% Load and set parameters

p = OmegaConf.load('params.yml')

# adjust the number of images to apply the model to;
# p.batch_size*p.num_batches first images will be selected from the test set
# specified in the split files (see 'splits' folder)
p.batch_size = 32
p.num_batches = 4

GT, pred_adv, pred_clean, image_filenames, images_clean, noise_adv = test_adversarial_attack(p, \
    compute_clean_predictions = True, return_images = True)

    
#%% Visualize the results

# select unique persons
_, img_indices = np.unique([fname.split('_')[0] for fname in image_filenames], return_index = True)
img_indices = np.sort(img_indices).astype(int)

# visualize clean and adversarial images, with the corresponding ground truth and predictions
batch_size = 4      # the number images to visualize in one plot
for batch_idx in range(img_indices.size // 4):
    batch_indices = img_indices[(batch_idx*batch_size):((batch_idx+1)*batch_size)]
    plot_batch(p, np.array(image_filenames)[batch_indices],
               images_clean[batch_indices, ...],
               (images_clean + noise_adv)[batch_indices, ...],
               GT[batch_indices, ...],
               pred_clean[batch_indices, ...],
               pred_adv[batch_indices, ...])