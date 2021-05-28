"""
Helper functions.
"""

import numpy as np

from sklearn import metrics

import matplotlib.pyplot as plt


def label2vector_CXR14(label):
    """
    Compute a binary class vector from a text label.

    """
    
    labels = ['No Finding', 'Cardiomegaly', 'Emphysema', 'Edema', 'Hernia', 'Pneumothorax', 'Effusion', 'Mass',
              'Fibrosis', 'Atelectasis', 'Consolidation', 'Pleural_Thickening', 'Nodule', 'Pneumonia', 'Infiltration']
              
    vec = np.zeros(15, dtype = int)
    
    for fName in label.split('|'):
        
        idx = labels.index(fName)
        
        vec[idx] = 1
            
    return labels, vec

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def roc_auc(GT, pred):
    """
    Compute the area under the ROC curve for all classes.

    """
    
    aucs = []
    
    for c in range(GT.shape[1]):
        fpr, tpr, thresholds = metrics.roc_curve(GT[:, c], pred[:, c])
        aucs.append(metrics.auc(fpr, tpr))
        
    return aucs

def show_batch(GT, pred_clean, pred_adv):
    """
    Print adversarial attack results.

    """
    
    for sample_idx in range(GT.shape[0]):
        
        print('GT :\n', GT[sample_idx, ...])
        print('pred_clean :\n', pred_clean[sample_idx, ...])
        print('pred_adv :\n', pred_adv[sample_idx, ...])

        print('GT_classes : ', np.where(GT[sample_idx, ...])[0])
        print('top_clean_classes : ', np.argsort(pred_clean[sample_idx, ...])[::-1][:2])
        print('top_adv_classes : ', np.argsort(pred_adv[sample_idx, ...])[::-1][:2])

def plot_batch(p, image_filenames, images_clean, images_adv,
               GT, pred_clean, pred_adv):
    """
    Plot adversarial attack results.

    """

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'brown', 'olive', 'gray', 'lawngreen', 'indigo', 'lavender', 'darkred']

    fig, axs = plt.subplots(len(image_filenames), 5)
    
    for imidx in range(len(image_filenames)):
        if imidx == 0:
            for j, title in zip(range(5), ['clean', 'adv', 'GT', 'clean', 'adv']):
                axs[imidx, j].set_title(title)
        axs[imidx, 0].imshow(images_clean[imidx, 0, ...])
        axs[imidx, 1].imshow(images_adv[imidx, 0, ...])
        barlist1 = axs[imidx, 2].bar(range(GT.shape[1]), GT[imidx, :])
        barlist2 = axs[imidx, 3].bar(range(GT.shape[1]), pred_clean[imidx, :])
        barlist3 = axs[imidx, 4].bar(range(GT.shape[1]), pred_adv[imidx, :])

        for barlist in [barlist1, barlist2, barlist3]:
            for idx, b in enumerate(barlist):
    	        barlist[idx].set_color(colors[idx])
                
        for j in [2, 3, 4]:
            axs[imidx, j].set_ylim(0, 1)
        
    fig.tight_layout()