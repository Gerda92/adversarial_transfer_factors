"""
Transform tables computed during experiments (by 'compute_table_X.py' files)
into tables shown in the manuscript and the supplement
by sorting / aggregating / rounding values.
"""

import os

from omegaconf import OmegaConf

import numpy as np
import pandas as pd


def pivot_df(df, grouping_columns, columns, values):
    """
    Turn table's columns into rows.

    """
    
    groupby_obj = df.groupby(grouping_columns)
    grouped_df = groupby_obj.size().reset_index()
    
    grouper = np.zeros((len(df.index)), int)
    
    for name, group in groupby_obj:
        grouper[group.index] = np.max(grouper) + 1
        
    new_df = df.pivot_table(index = grouper, columns = columns,
                            values = values).reset_index()
    
    return pd.concat([grouped_df.iloc[:, :-1], \
                      new_df.iloc[:, 1:]], axis = 1)


#%% Load and specify parameters

p = OmegaConf.load('base_params.yml')


#%% Print and save Table 2 (perturbation degree, FGSM vs PGD)
#   and the corresponding tables in the supplement.

table1 = pd.read_csv(os.path.join(p.results_path, 'Table_1.csv'))

# remove clean performance rows
table1 = table1.loc[table1.surrogate != '-', :]
table1.index = range(len(table1.index))

# round, make 'attack method' and 'epsilon' columns, sort
table1_suppl = table1.round({'mean_auc': 2})
table1_suppl = pivot_df(table1_suppl, ['target', 'surrogate', 'shuffle'], ['attack_method', 'epsilon'], 'mean_auc')
table1_suppl = table1_suppl.sort_values( ['target', 'surrogate'], ascending = False)

# print and save
print(table1_suppl)
table1_suppl.to_csv(os.path.join(p.results_path, 'Table_1_supplement.csv'), index = False)

# average among target-surrogate architecture pairs, round
# (makes the manuscript version of the table)
table1_manu = table1.groupby(by = \
    [table1.attack_method, table1.epsilon, table1.shuffle]).mean().reset_index()
table1_manu = table1_manu.round({'mean_auc': 2})
table1_manu = pivot_df(table1_manu, ['shuffle'], ['attack_method', 'epsilon'], 'mean_auc')

# print and save
print(table1_manu)
table1_manu.to_csv(os.path.join(p.results_path, 'Table_1_manuscript.csv'), index = False)


#%% Print and save Table 3 (pre-training vs random initialization)
#   and the corresponding tables in the supplement.

table2 = pd.read_csv(os.path.join(p.results_path, 'Table_2.csv'))
table2 = table2.iloc[:, 1:]

# save clean performance values to a different data frame
table2_clean = table2.loc[table2.surrogate == '-', :]
table2_clean.index = range(len(table2_clean.index))

# remove clean performance rows
table2 = table2.loc[table2.surrogate != '-', :]
table2.index = range(len(table2.index))

# round, make 'attack method' columns, sort
table2_suppl = table2.round({'mean_auc': 2, 'perc_clean': 0})
table2_suppl = pivot_df(table2_suppl, ['target', 'target_pretrained', 'surrogate', 'surrogate_pretrained'], 'attack_method', ['mean_auc', 'perc_clean'])
table2_suppl = table2_suppl.sort_values(
    ['target', 'target_pretrained', 'surrogate', 'surrogate_pretrained'], ascending = False)

# print and save
print(table2_suppl)
table2_suppl.to_csv(os.path.join(p.results_path, 'Table_2_supplement.csv'), index = False)

# average
architecture = table2.target == table2.surrogate
architecture.name = 'same_architecture'
table2_manu = table2.groupby(by = \
    [architecture, table2.target_pretrained, table2.surrogate_pretrained]).mean()
table2_manu = table2_manu.round({'mean_auc': 2, 'perc_clean': 0})
table2_manu = table2_manu.sort_values(
    ['same_architecture', 'target_pretrained', 'surrogate_pretrained'],
    ascending = False)

# print and save
print(table2_manu)
table2_manu.to_csv(os.path.join(p.results_path, 'Table_2_manuscript.csv'))


#%% Print and save a version of Table 3 comparing between architectures

# average
table2_manu2 = table2.groupby(by = \
    [table2.target, table2.surrogate]).mean()
table2_manu2 = table2_manu2.round({'mean_auc': 2, 'perc_clean': 0})
table2_manu2 = table2_manu2.sort_values(
    ['target', 'surrogate'],
    ascending = False)

# print and save
print(table2_manu2)
table2_manu2.to_csv(os.path.join(p.results_path, 'Table_2_architecture.csv'))


#%% Print and save clean performances for Table 3 (pre-training vs random initialization)

table2_clean_suppl = table2_clean.round({'mean_auc': 2, 'perc_clean': 0})
table2_clean_suppl = table2_clean_suppl.sort_values(
    ['target', 'target_pretrained'], ascending = False)

# print and save
print(table2_clean_suppl)
table2_clean_suppl.to_csv(os.path.join(p.results_path, 'Table_2_clean_supplement.csv'), index = False)

# average
table2_clean_manu = table2_clean.groupby(by = \
    [table2_clean.target_pretrained]).mean()
table2_clean_manu = table2_clean_manu.round({'mean_auc': 2, 'perc_clean': 0})
table2_clean_manu = table2_clean_manu.sort_values(
    ['target_pretrained'], ascending = False)

# print and save
print(table2_clean_manu)
table2_clean_manu.to_csv(os.path.join(p.results_path, 'Table_2_clean_manuscript.csv'))


#%% Print and save Table 4 (training data assymetry)
#   and the corresponding tables in the supplement.

table3 = pd.read_csv(os.path.join(p.results_path, 'Table_3.csv'))
table3 = table3.iloc[:, 1:]

# remove clean performance rows
table3 = table3.loc[table3.surrogate != '-', :]
table3.index = range(len(table3.index))

# round, make 'attack method' columns, sort
table3_suppl = table3.round({'mean_auc': 2, 'perc_clean': 0})
table3_suppl = pivot_df(table3_suppl, ['target', 'surrogate', 'surrogate_train_set'], 'attack_method', ['mean_auc', 'perc_clean'])
table3_suppl = table3_suppl.sort_values(['target', 'surrogate'], ascending = False)

# print and save
print(table3_suppl)
table3_suppl.to_csv(os.path.join(p.results_path, 'Table_3_supplement.csv'), index = False)

architecture = table3.target == table3.surrogate
architecture.name = 'same_architecture'
table3_manu = table3.groupby(by = \
    [architecture, table3.surrogate_train_set]).mean()
table3_manu = table3_manu.round({'mean_auc': 2, 'perc_clean': 0})
table3_manu = table3_manu.sort_values(['same_architecture'],
                                            ascending = False)

# print and save
print(table3_manu)
table3_manu.to_csv(os.path.join(p.results_path, 'Table_3_manuscript.csv'))


#%% Print and save a version of Table 4 comparing between architectures

# average
table3_manu2 = table3.groupby(by = \
    [table3.target, table3.surrogate, table3.surrogate_train_set]).mean()
table3_manu2 = table3_manu2.round({'mean_auc': 2, 'perc_clean': 0})
table3_manu2 = table3_manu2.sort_values(
    ['surrogate_train_set', 'target', 'surrogate'],
    ascending = [True, False, False])

# print and save
print(table3_manu2)
table3_manu2.to_csv(os.path.join(p.results_path, 'Table_3_architecture.csv'))


#%% Print and save Table 5 (pre-training, training data assymetry)
#   and the corresponding tables in the supplement.

table4 = pd.read_csv(os.path.join(p.results_path, 'Table_4.csv'))
table4 = table4.iloc[:, 1:]

# save clean performance values to a different data frame
table4_clean = table4.loc[table4.surrogate == '-', :]
table4_clean.index = range(len(table4_clean.index))

# remove clean performance rows
table4 = table4.loc[table4.surrogate != '-', :]
table4.index = range(len(table4.index))

# round, make 'attack method' columns, sort
table4_suppl = table4.round({'mean_auc': 2, 'perc_clean': 0})
table4_suppl = pivot_df(table4_suppl, ['target', 'target_pretrained', 'surrogate', 'surrogate_pretrained'], 'attack_method', ['mean_auc', 'perc_clean'])
table4_suppl = table4_suppl.sort_values(['target', 'target_pretrained', 'surrogate', 'surrogate_pretrained'], ascending = False)

# print and save
print(table4_suppl)
table4_suppl.to_csv(os.path.join(p.results_path, 'Table_4_supplement.csv'), index = False)

# average
architecture = table4.target == table4.surrogate
architecture.name = 'same_architecture'
table4_manu = table4.groupby(by = \
    [architecture, table4.target_pretrained, table4.surrogate_pretrained]).mean()
table4_manu = table4_manu.round({'mean_auc': 2, 'perc_clean': 0})
table4_manu = table4_manu.sort_values(
    ['same_architecture', 'target_pretrained', 'surrogate_pretrained'],
    ascending = False)

# print and save
print(table4_manu)
table4_manu.to_csv(os.path.join(p.results_path, 'Table_4_manuscript.csv'))


#%% Print and save clean performances for Table 4 (pre-training vs random initialization)

# round, make 'attack method' columns, sort
table4_clean_suppl = table4_clean.round({'mean_auc': 2, 'perc_clean': 0})
table4_clean_suppl = pivot_df(table4_clean_suppl, ['target', 'target_pretrained'], 'attack_method', ['mean_auc', 'perc_clean'])
table4_clean_suppl = table4_clean_suppl.sort_values(
    ['target', 'target_pretrained'], ascending = False)

# print and save
print(table4_clean_suppl)
table4_clean_suppl.to_csv(os.path.join(p.results_path, 'Table_4_clean_supplement.csv'), index = False)

# average
table4_clean_manu = table4_clean.groupby(by = \
    [table4_clean.target_pretrained]).mean()
table4_clean_manu = table4_clean_manu.round({'mean_auc': 2, 'perc_clean': 0})
table4_clean_manu = table4_clean_manu.sort_values(
    ['target_pretrained'], ascending = False)

# print and save
print(table4_clean_manu)
table4_clean_manu.to_csv(os.path.join(p.results_path, 'Table_4_clean_manuscript.csv'))


#%% Print and save a version of Table 5 comparing between architectures

# average
table4_manu2 = table4.groupby(by = \
    [table4.target, table4.surrogate]).mean()
table4_manu2 = table4_manu2.round({'mean_auc': 2, 'perc_clean': 0})
table4_manu2 = table4_manu2.sort_values(
    ['target', 'surrogate'],
    ascending = False)

# print and save
print(table4_manu2)
table4_manu2.to_csv(os.path.join(p.results_path, 'Table_4_architecture.csv'))