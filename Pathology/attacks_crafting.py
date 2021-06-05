'''
Functions to craft adversarial examples given a surrogate model and an attack configuration, as well as to generate control noise.

@author: Suzanne Wetstein
'''

import keras
from keras.models import Model
from keras import backend as K
import tensorflow as tf

import numpy as np
import os
import h5py
from keras.utils import HDF5Matrix

# ADVERSARIAL LOSS FUNCTIONS

def classification(model, GT):
    # GT: target for the adversarial attack
    pred = model.outputs[-1]
    return tf.keras.backend.binary_crossentropy(GT, pred)

def initFGSM(model, obj):
    # initialize a gradient calculating function for FGSM
    # model: tensorflow model
    # obj: objective function; see targetSegmentation for an example
    GT = K.placeholder()
    grad = K.sign(tf.gradients(obj(model, GT), model.layers[0].input)[0])
    noiseFun = K.function([K.learning_phase(), model.layers[0].input, GT], [grad])
    return {'noiseFun': noiseFun}    


def iFGSM(funs, X, Y, epsilon = 0.1, alpha = 0.01, steps = 20):
    # compute an adversarial example
    # funs: output of initFGSM
    # X: network input
    # Y: target output
    # epsilon: the maximum amount of noise
    # alpha and steps: step size for the optimization and the number of steps
    # The latter two highly depend on the range of your output!
    noiseFun = funs['noiseFun']
    res = X
    for i in range(steps): 
        res = np.clip(np.minimum(X + epsilon, np.maximum(res + noiseFun([False, res, Y])[0]*alpha, X - epsilon)), -1, 1)
    return res

###############
    
def craft_attack(im, label, surrogate_model, attack_method, attack_config, config_dict):
    #Set attack config
    steps = config_dict['steps']
    epsilon = config_dict['epsilon']
    alpha = config_dict['alpha']
    
    print('Performing ', attack_method, ' attack')
    print('epsilon: ', epsilon)
    print('alpha: ', alpha)
    print('steps: ', steps)

    #Get categorical label
    y = np.reshape(label, (-1,1))

    advFuns = initFGSM(surrogate_model, classification)

    #Craft attack
    im_adv = iFGSM(advFuns, im[np.newaxis], y[np.newaxis], epsilon, alpha, steps)
    
    print('Attack crafted!')
    
    return im_adv

#Control noise functions

def shuffle_noise(noise):
    flatNoise = np.ndarray.flatten(noise)
    np.random.shuffle(flatNoise)
    permuted = flatNoise.reshape(noise.shape)
    return permuted

def generate_control_noise(im, im_adv):
    noise = im_adv-im
    noise_shuffle = shuffle_noise(noise)
    im_control_noise = np.clip(im+noise_shuffle, -1, 1)
    return im_control_noise

def get_adv_noise(im, im_adv):
    noise = im_adv-im
    stretch_noise = (noise - np.min(noise))/(np.max(noise)-np.min(noise))
    return stretch_noise