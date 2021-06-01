

import os

from omegaconf import OmegaConf

import numpy as np
import pandas as pd

import helpers as h
from compute_attacks import test_adversarial_attack


#%% Load and specify parameters

p = OmegaConf.load('base_params.yml')

p.batch_size = 32
p.num_batches = 800

architectures = ['InceptionV3', 'DenseNet121']
train_sets = ['d1', 'd2', 'd3']
attack_methods = ['FGSM', 'PGD']


#%% Compute figures for Table 3

# data frames for saving the results
table = pd.DataFrame(columns = ['target', 'surrogate', 'surrogate_train_set', 'attack_method', 'mean_auc', 'perc_clean'])

# shared experiment parameters
p.attack_type = 'black_box'
p.target.pretrained = p.surrogate.pretrained = False
p.target.train_set = 'd1'
p.attack.epsilon = 0.02

# iterate throgh all experiment parameters
for p.target.keras_arch in architectures:
    
    # compute predictions on clean examples:
    GT, pred_clean = test_adversarial_attack(p, compute_clean_predictions = 'clean_only')
    auc_clean = np.mean(h.roc_auc(GT, pred_clean))

    # add a row to the table
    table.loc[len(table.index)] = [p.target.keras_arch, \
                                   '-', '-', 'None', auc_clean, 100]
    table.to_csv(os.path.join(p.results_path, 'Table_3.csv'))

    GT, pred_clean_prev = h.load_previous_results(p, clean = True)
    auc_clean_prev = np.mean(h.roc_auc(GT, pred_clean_prev))


    for p.surrogate.keras_arch in architectures:
        for p.surrogate.train_set in train_sets:
            for attack in attack_methods:

                if p.target.keras_arch == p.surrogate.keras_arch and p.target.train_set == p.surrogate.train_set:
                    p.surrogate.instance = 'v2'
                else:
                    p.surrogate.instance = 'v1'
                
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
                table.loc[len(table.index)] = [p.target.keras_arch, \
                                               p.surrogate.keras_arch, p.surrogate.train_set, attack, auc, perc]
                table.to_csv(os.path.join(p.results_path, 'Table_3.csv'))
