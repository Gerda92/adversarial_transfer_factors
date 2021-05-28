"""
Functions to build target and surrogate models.
"""

from tensorflow import keras

def get_keras_net(arch_name, p):
    """
    Builds an architecture based on keras.applications implementation,
    specified by name arch_name and parameters p.

    """

    # ChestX-Ray14 images are black and white, whereas most architectures
    # require 3-channel images. We thus prepend additional layers in front
    # of the architecture layers: an input layer of shape 1 x img_res_x x img_res_y
    # followed by a tiling layer that replicates the input wrt the channel
    # dimension thrice, providing 3 x img_res_x x img_res_y output.
    inp_layer = keras.layers.Input(shape = [1] + list(p.input_res))
    tile = keras.layers.concatenate([inp_layer]*3, 1)

    base_model = getattr(keras.applications, arch_name)(include_top = False,    # strip the final layers
        input_tensor = tile)

    # add necessary pooling and classification layers
    last = base_model.output    
    last = keras.layers.GlobalAveragePooling2D()(last)    
    output = keras.layers.Dense(p.num_output_maps, activation = p.final_activation)(last)
    
    model = keras.models.Model(inputs = [inp_layer], outputs = [output])
    
    return model