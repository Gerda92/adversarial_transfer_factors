# Adversarial attacks on color fundus images (Kaggle DR Detection dataset)

## Software prerequisites

The code requires TensorFlow 1.12 and other python packages listed in **lib_versions.txt**.
The code was written and tested in Python 3.

## Other prerequisites

The Kaggle Diabetic Retinopathy Detection dataset can be downloaded [here](https://www.kaggle.com/c/diabetic-retinopathy-detection/data).

The Jupyter notebook called **Main.ipynb** contains the code for crafting and testing black-box attacks corresponding to the Ophthalmology application in the publication. The notebook is structured as follows:

**1) Select and preprocess original image.** Load original image and create prprocessed 512x512 RGB image, which is the expected input to the trained target and surrogate models.

**2) Load surrogate model.** Load surrogate model that will be used to craft the adversarial attacks in the black-box setting.

**3) Craft adversarial attack.** Define attack configuration and craft the attack using the surrogate model.

**4) Load target model.** Load target model that will be attacked in the black-box setting. 

**5) Attack target model.** Obtain predictions in clean and adversarial setting. Visualization of original image, adversarial image, and corresponding adversarial noise.

The notebook makes use of the folder structure and files of the **data repository** that can be downloaded [here](link). In that data repository:

The **metadata** folder contains:
- CSV file (*kaggle_all_array_labels_bin.csv*) with binarized labels (non-referable DR = 0 (stages 0 and 1) and referable DR = 1 (stages 2 to 4)) for all images in the [Kaggle DR dataset](https://www.kaggle.com/c/diabetic-retinopathy-detection). 
- CSV files with the filenames of the images included in each data partition used in the paper:
    - Development sets: *d1*, *d2*, and *d3* (equivalent top *d2/2*), and the corresponding filenames used for training (*t1*, *t2*, *t3*) and validation (*v1*, *v2*, *v3*) in each development set.
    - Test set

The **data** folder contains 2 example images (one with referable DR label and one with non-referable DR label) from the [Kaggle DR dataset](https://www.kaggle.com/c/diabetic-retinopathy-detection) included in the test set from the publication. 

The **models** folder contains all the models trained for the publication that served as target or surrogate models in the black-box adversarial settings, plus the optimal referable/non-referable thresholds computed from the corresponding validation set (threshold that maximizes sensitivity and specificity).
