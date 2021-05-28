"""
Functions for computing adversarial perturbations.
"""

import tensorflow as tf
from tensorflow.keras import backend as K

import numpy as np


def CE(model, GT):
    """
    Cross-entropy loss function operating on logits.

    """
    pred = model.outputs[-1]
    return K.mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = pred, labels = GT))

def initFGSM(model, loss_func_name):
    """
    Instantiates a computation graph for a function computing
    sign of the gradient of the loss_func w.r.t. model input.
    """

    loss_func = globals()[loss_func_name]
    
    GT = tf.placeholder('float32')
    
    grad = tf.keras.backend.sign(tf.gradients(loss_func(model, GT), model.layers[0].input)[0])
    
    sign_of_gradient_func = tf.keras.backend.function([tf.keras.backend.learning_phase(), model.layers[0].input, GT], [grad])

    return sign_of_gradient_func

def iFGSM(sign_of_gradient_func, X, Y, epsilon, alpha = 0.01, steps = 20):
    """
    Computes a iterative FGSM (PGD) attack.

    Parameters
    ----------
    sign_of_gradient_func : TF function
        Output of initFGSM().
    X : a numpy array
        An input image.
    Y : a numpy array
        A ground truth label.
    epsilon : float
        Maximum perturbation degree.
    alpha : float, optional
        Step size. The default is 0.01.
    steps : int, optional
        The number of iterations. The default is 20.

    Returns
    -------
    A numpy array
        Adversarial perturbation.

    """
    
    res = X.copy()
    
    for i in range(steps):
        
        res = np.clip(res + sign_of_gradient_func([False, res, Y])[0]*alpha, X - epsilon, X + epsilon)

    res = np.clip(res, np.min(X), np.max(X))
    
    return res - X