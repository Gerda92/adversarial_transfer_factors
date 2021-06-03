
import os

from omegaconf import OmegaConf

import numpy as np
import pandas as pd

import helpers as h
from compute_attacks import test_adversarial_attack


#%% Load and specify parameters

p = OmegaConf.load('base_params.yml')

# this will apply the attacks to all 25596 test images:
p.batch_size = 32
p.num_batches = 800

architectures = ['InceptionV3', 'DenseNet121']
attack_methods = ['FGSM', 'PGD']


#%% Compute figures for Table 3 in the manuscript

# data frames for saving the results
table = pd.DataFrame(columns = ['target', 'target_pretrained', 'surrogate', 'surrogate_pretrained', \
    'attack_method', 'mean_auc', 'perc_clean'])

# shared experiment parameters
p.attack_type = 'black_box'
p.target.train_set = 'd0'
p.surrogate.train_set = 'd2'
p.surrogate.instance = 'v1'
p.attack.epsilon = 0.02

# iterate throgh all experiment parameters
for p.target.keras_arch in architectures:
    for p.target.pretrained in [True, False]:

        # compute predictions on clean examples:
        GT, pred_clean = test_adversarial_attack(p, compute_clean_predictions = 'clean_only')
        auc_clean = np.mean(h.roc_auc(GT, pred_clean))

        # add a row to the table
        table.loc[len(table.index)] = \
            [p.target.keras_arch, p.target.pretrained,
             '-', '-', 'None', auc_clean, 100]
        table.to_csv(os.path.join(p.results_path, 'Table_4.csv'))
                    
        
        for p.surrogate.pretrained in [True, False]:
            for p.surrogate.keras_arch in architectures:
                for attack in attack_methods:
                    
                    if attack == 'FGSM':
                        p.attack.alpha = p.attack.epsilon
                        p.attack.num_steps = 1
                    else:
                        p.attack.alpha = 0.01
                        p.attack.num_steps = 20
                        
                    GT, pred_adv = test_adversarial_attack(p)
                    
                    auc = np.mean(h.roc_auc(GT, pred_adv))
                    perc = auc / auc_clean * 100
    
                    # add a row to the table
                    table.loc[len(table.index)] = \
                        [p.target.keras_arch, p.target.pretrained,
                         p.surrogate.keras_arch, p.surrogate.pretrained, attack, auc, perc]
                    table.to_csv(os.path.join(p.results_path, 'Table_4.csv'))
