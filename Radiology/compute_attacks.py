
import os, yaml, time

import numpy as np
import pandas as pd

import tensorflow as tf

from tqdm import trange

import helpers as h
import improc
from build_model import get_keras_net
from adv_attack_func import initFGSM, iFGSM

# set dimension order for tensorflow
tf.keras.backend.set_image_data_format('channels_first')


def test_adversarial_attack(p, compute_clean_predictions = False, return_images = False):
    """
    Given experiment parameters, compute adversarial examples using a surrogate network
    and test a target network on them.

    Parameters
    ----------
    p : OmegaConf object
        Parameters of the attack (see params.yml for an example).
    compute_clean_predictions : bool or str
        Whether to compute predictions on 'clean' (original, non-adversarial) images.
        Setting to 'clean_only' will only compute predictions on clean images. Default is False.
    return_images: bool
        Whether to return original and adversarial images. Default is False.

    Returns
    -------
    Ground truth labels and predictions on adversarial test images;
    optionally, predictions on clean images and/or original images and adversarial noise.

    """

#%% Preprocess parameters

    # if using a white-box attack, surrogate model is the same as the target
    if p.attack_type == 'white_box':
        p.surrogate = p.target

#%% Load a meta dataset with image names and labels
#   (supplied with the ChestX-Ray14 dataset)

    data_meta = pd.read_csv(os.path.join(p.data_path, 'Data_Entry_2017.csv'))

#%% Load a train-validation-test split

    # load indices of train, validation, and test set images
    # corresponding to rows in 'Data_Entry_2017.csv' file
    with open('splits/%s_indices_data_entry.yml' % p.target.train_set, 'r') as stream:
        split_indices = yaml.safe_load(stream)
    
    # restrict the number of batches to process to the number of batches covering the the test set
    p.batch_num = min(p.num_batches, int(np.ceil(len(split_indices['testIDs']) / p.batch_size)))

#%% Compute names of the target and surrogate (for loading them)

    target_model_name = '%s_%s_%s' % (p.target.keras_arch, 'ImageNet' if p.target.pretrained else 'random',
                                          p.target.train_set)
    
    surrogate_model_name = '%s_%s%s_%s' % (p.surrogate.keras_arch, 'ImageNet' if p.surrogate.pretrained else 'random',
                                          '_v2' if p.surrogate.instance == 'v2' else '',
                                          p.surrogate.train_set)

#%% Load the surrogate

    if compute_clean_predictions != 'clean_only':

        print('Loading surrogate model', surrogate_model_name, '...')
    
        surrogate_model = get_keras_net(p.surrogate.keras_arch, p.arch_params)
        surrogate_model.load_weights(os.path.join(p.model_path, '%s.hdf5' % surrogate_model_name))

#%% Instantiate a graph for computing adversarial attacks

        start = time.time()
        sign_of_gradient_func = initFGSM(surrogate_model, p.attack.loss)
        print('Compiling the gradient sign function: ', time.time() - start, ' sec.')

#%% Compute adversarial perturbations

    noise_adv = np.zeros([np.minimum(p.batch_size*p.num_batches, len(split_indices['testIDs'])), 1] \
                         + list(p.arch_params.input_res), dtype = np.float32)

    if compute_clean_predictions != 'clean_only':
        
        print('Computing attacks:')
    
        for batch_idx in trange(p.num_batches):
    
            images_clean, GT, text_labels = improc.load_batch_CXR14(p,
                split_indices['testIDs'][(batch_idx*p.batch_size):((batch_idx+1)*p.batch_size)],
                data_meta)
    
            noise_adv_batch = iFGSM(sign_of_gradient_func, images_clean, GT, p.attack.epsilon, p.attack.alpha, p.attack.num_steps)
            if p.attack.shuffle_noise:
                noise_adv_batch = improc.shuffle_array(noise_adv_batch)
            noise_adv[(batch_idx*p.batch_size):((batch_idx+1)*p.batch_size)] = noise_adv_batch

#%% Release the surrogate model and load the target

    tf.keras.backend.clear_session()
    
    print('\nLoading target model', target_model_name, '...')
    
    target_model = get_keras_net(p.target.keras_arch, p.arch_params)
    target_model.load_weights(os.path.join(p.model_path, '%s.hdf5' % target_model_name))

#%% Apply the adversarial noise

    if return_images:
        images_clean_all = np.zeros(noise_adv.shape, dtype = np.float32)

    GT_all = np.zeros((np.minimum(p.batch_size*p.num_batches, len(split_indices['testIDs'])), 15), dtype = int)
    pred_clean = np.zeros(GT_all.shape)
    pred_adv = np.zeros(GT_all.shape)
    
    print('Applying attacks:')

    for batch_idx in trange(p.num_batches):
        images_clean, GT, text_labels = improc.load_batch_CXR14(p,
            split_indices['testIDs'][(batch_idx*p.batch_size):((batch_idx+1)*p.batch_size)],
            data_meta)
        
        if return_images:
            images_clean_all[(batch_idx*p.batch_size):((batch_idx+1)*p.batch_size), ...] = images_clean
            
        GT_all[(batch_idx*p.batch_size):((batch_idx+1)*p.batch_size)] = GT

        images_adv = images_clean + noise_adv[(batch_idx*p.batch_size):((batch_idx+1)*p.batch_size)]

        if compute_clean_predictions:
            pred_clean[(batch_idx*p.batch_size):((batch_idx+1)*p.batch_size)] = \
                h.sigmoid(target_model.predict(images_clean, verbose = 0))

        if compute_clean_predictions != 'clean_only':
            pred_adv[(batch_idx*p.batch_size):((batch_idx+1)*p.batch_size)] = \
                h.sigmoid(target_model.predict(images_adv, verbose = 0))

#%% Assemble the results tuple

    result = [GT_all]
    
    if compute_clean_predictions != 'clean_only':
        result.append(pred_adv)   
    
    if compute_clean_predictions:
        result.append(pred_clean)
    
    if return_images:
        image_filenames = [data_meta.at[sample_index, 'Image Index']
                           for sample_index in split_indices['testIDs'][:GT_all.shape[0]]]
        result += [image_filenames, images_clean_all, noise_adv]
    
    return tuple(result)

