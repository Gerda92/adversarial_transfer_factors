"""
Functions to craft adversarial examples given a surrogate model and an attack configuration, as well as to generate control noise.

@author: CristinaGGonzalo
"""


import keras
from keras.models import Model
from keras import backend as K
from keras.utils import to_categorical
import tensorflow as tf


import os
import numpy as np
import math
from random import shuffle

import aux_functions as aux


### Advesarial loss functions

def classification(model, GT):
    # GT: target for the adversarial attack
    # if not softmaxed already
    pred = model.outputs[-1]
    return tf.keras.backend.categorical_crossentropy(GT, pred)

def initFGSM(model, obj):
    # initialize a gradient calculating function for FGSM
    # model: tensorflow model
    # obj: objective function; see targetSegmentation for an example
    GT = K.placeholder()
    grad = K.sign(tf.gradients(obj(model, GT), model.layers[0].input)[0])
    noiseFun = K.function([K.learning_phase(), model.layers[0].input, GT], [grad])
    return {'noiseFun': noiseFun}


def clip_between_bounds(images, lowerbound, upperbound):
    '''
    returns images, within lower and upperbound, and clipped within -1, 1
    '''
    res_0 = np.minimum(images, upperbound)
    res_1 = np.maximum(res_0, lowerbound)
    res_clip = np.clip(res_1, -1, 1)
    return res_clip


def iFGSM(funs, X, Y, epsilon = 0.1, alpha = 0.01, steps = 20):
    # compute an adversarial example
    # funs: output of initFGSM
    # X: network input
    # Y: target output
    # epsilon: the maximum amount of noise
    # alpha and steps: step size for the optimization and the number of steps
    noiseFun = funs['noiseFun']
    learning_phase = False
    res = X
    #Noise boundaries
    lowerbound = X - epsilon
    upperbound = X + epsilon
    
    for i in range(steps):
        noise_output = noiseFun([learning_phase, res, Y])[0]
        modified_image = res + noise_output * alpha
        res = clip_between_bounds(modified_image, lowerbound, upperbound)
         
    return res

##############

def craft_attack (img, label, surrogate_model, attack_method, attack_config, config_dict):
    #Set attack config
    steps = config_dict['steps']
    epsilon = config_dict['epsilon']
    alpha = config_dict['alpha']
    
    print('Performing ', attack_method, ' attack')
    print('epsilon: ', epsilon)
    print('alpha: ', alpha)
    print('steps: ', steps)
    
    #Normalize image (-1 to 1)
    x = aux.normalize(img)

    #Get categorical label
    y = to_categorical(label, 2)

    advFuns = initFGSM(surrogate_model, classification)

    #Craft attack
    adversarial_output = iFGSM(advFuns, x[np.newaxis], y[np.newaxis], epsilon, alpha, steps)
    
    #Denormalize adv image (0 to 255)
    img_adv = aux.denormalize(np.squeeze(adversarial_output))
    
    print('Attack crafted!')
    
    return img_adv



### Control noise functions
def shuffle_noise(noise):
    flat_noise = noise.flatten()
    shuffle(flat_noise)
    return np.reshape(flat_noise, noise.shape)


def generate_control_noise(img, img_adv):
    #Extract adversarial noise
    noise = aux.to_float(img_adv) - aux.to_float(img)
    
    #Shuffle noise and add to orig image
    noise_shuffle = shuffle_noise(noise)
    img_control_noise = aux.to_uint8(aux.to_float(img) + noise_shuffle)
    
    return img_control_noise