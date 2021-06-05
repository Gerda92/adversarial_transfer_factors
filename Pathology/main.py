'''Adversarial Attack Vulnerability of Medical Image Analysis Systems: Unexplored Factors
G. Bortsova, C. González-Gonzalo, S. Wetstein, F. Dubost, I. Katramados, L. Hogeweg, B. Liefers, B. van Ginneken, J. Pluim, M. Veta, C. Sánchez, M. de Bruijne. "Adversarial Attack Vulnerability of Medical Image Analysis Systems: Unexplored Factors", Medical Image Analysis, 2021.

Pathology
This notebook contains the code for crafting and testing black-box attacks corresponding to the Pathology application: PatchCamelyon (PCAM) metastatic tissue detection in histopathologic scans of lymph node sections.

The notebook makes use of the folder structure and some of the files in the data repository available at: add Zenodo link. In that repository:

The metadata folder contains:

CSV files with the filenames of the images included in the additional data partitions (d1/10 and d2/2) used in the paper:
Development set: d110 (equivalent to d1/10), and the corresponding filename used for training (t110) and validation (v110) for this development set.
Development set: d22 (equivalent to d2/2), and the corresponding filename used for training (t22) and validation (v22) for this development set.
'''

metadata_folder = 'metadata/'

'''The data folder contains 2 example images (one with metastatic tissue present and one without) from the PCam dataset included in the test set from the publication.'''

data_folder = 'data/'

'''The models folder contains all the models trained for the publication that served as target or surrogate models in the black-box adversarial settings.'''

models_folder = 'models/'

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model

from attacks_crafting import craft_attack, generate_control_noise, get_adv_noise

'''
1) Select and preprocess original image
'''

im = np.asarray(Image.open(os.path.join(data_folder, 'original_positive.jpg')))
label = 1

#Preprocess the image
im = im/255
im -= 0.5
im *= 2

'''
2) Load the surrogate model
    This is the model that will be used to craft the attacks in the black-box setting. These are the different architectures, development sets, and initialization configurations that were used for the surrogate models in the different experiments of the publication.

    Architecture:
        'IV3'
        'DenseNet121'
    Dev_set:
        'd1'
        'd1_bis' (for black-box setting with same architecture)
        'd2'
        'd22' (equivalent to d2/2)
    Config:
        'pretrained' (model initialized with ImageNet pretraining)
        'heinit' (model randomly initialized with He initializer)
'''

#Define here chosen surrogate configuration and load model:
surrogate_architecture = 'IV3'
surrogate_dev_set = 'd1'
surrogate_config = 'pretrained'

surrogate_folder = os.path.join(models_folder, surrogate_architecture, surrogate_config+'_'+surrogate_dev_set)
print(surrogate_folder)
surrogate_model = load_model(os.path.join(surrogate_folder, 'pat_model.h5'))
surrogate_model.summary()

'''
3) Craft adversarial attack
Define attack configuration
These are the different attack configurations used in the different black-box adversarial settings of the publication.

FGSM:
Config1: 1 step, alpha 0.01, epsilon 0.01
Config2: 1 step, alpha 0.02, epsilon 0.02
Config3: 1 step, alpha 0.03, epsilon 0.03
Config4: 1 step, alpha 0.04, epsilon 0.04
Config5: 1 step, alpha 0.05, epsilon 0.05
Config6: 1 step, alpha 0.06, epsilon 0.06

PGD:
Config1: 20 steps, alpha 0.01, epsilon 0.01
Config2: 20 steps, alpha 0.01, epsilon 0.02
Config3: 20 steps, alpha 0.01, epsilon 0.03
Config4: 20 steps, alpha 0.01, epsilon 0.04
Config5: 20 steps, alpha 0.01, epsilon 0.05
Config6: 20 steps, alpha 0.01, epsilon 0.06

In order to generate the control noise version (adversarial noise spatially shuffled), set the variable control_noise to True.
'''

#Select attack configuration:    
attack_method = 'FGSM'
attack_config = 'config2'
config_dict = {"steps": 1, "alpha": 0.02, "epsilon": 0.02}
control_noise = True

#Craft attack
im_adv = craft_attack(im, label, surrogate_model, attack_method, attack_config, config_dict)
if control_noise:
    im_control_noise = generate_control_noise(im, im_adv)
    
'''
4) Load target model
   Load target model that will be attacked in the black-box setting. These are the different architectures, development sets, and initialization configurations that were used for the target models in the different experiments of the publication.

    Architecture:
        'IV3'
        'DenseNet121'
    Dev_set:
        'd1'
        'd110' (equivalent to d1/10)
    Config:
        'pretrained' (model initialized with ImageNet pretraining)
        'heinit' (model randomly initialized with He initializer)
'''

#Define here chosen target configuration and load model.
target_architecture = 'IV3'
target_dev_set = 'd1'
target_config = 'pretrained'

target_folder = os.path.join(models_folder, target_architecture, target_config+'_'+target_dev_set)
print(target_folder)
target_model = load_model(os.path.join(target_folder, 'pat_model.h5'))
target_model.summary()

'''
5) Attack target model
'''
#Obtain predictions in clean and adversarial setting
orig_pred = target_model.predict(im[np.newaxis])
orig_pred = orig_pred[:,1][0]

adv_pred = target_model.predict(im_adv[np.newaxis])
adv_pred = adv_pred[:,1][0]

print('Original label: ', label)
print('Clean prediction: ', orig_pred)
print('Adversarial prediction: ', adv_pred)

if control_noise:
    control_pred = target_model.predict(im_control_noise[np.newaxis])
    control_pred = control_pred[:,1][0]      
    print('Control noise prediction: ', control_pred)
    
#Get adversarial noise for visualization. The adversarial noise is equivalent to the difference between the original and the adversarial image.
plot_im = (im/2+0.5)*255
plot_im_adv = (im_adv/2+0.5)*255    
adv_noise = (get_adv_noise(plot_im,plot_im_adv)/2+0.5)*255
    
fig, axes = plt.subplots(1,3, figsize=(40, 40))
for ax in axes:
    ax.set_axis_off()
axes[0].imshow(plot_im)
axes[0].set_title('Original image', size=30)
axes[1].imshow(plot_im_adv)
axes[1].set_title('Adversarial image', size=30)
axes[2].imshow(adv_noise)
axes[2].set_title('Adversarial noise', size=30)
plt.show()   