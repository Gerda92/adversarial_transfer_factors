# Adversarial attacks on pathology image patches (PatchCamelyon dataset)

## Software prerequisites

The code requires TensorFlow 1 (tested in tf 1.4.0) and other python packages listed in **lib_versions.txt**.
The code was written and tested in Python 3.

## Other prerequisites

The PatchCamelyon dataset can be downloaded [here](https://github.com/basveeling/pcam).

## File structure

The python file called **main.py** contains the code for crafting and testing black-box attacks corresponding to the pathology application in the publication. The notebook is structured as follows:

**1) Select and preprocess original image.** Load original image and create normalized 96x96 pixel 3-channel image patch, which is the expected input to the trained target and surrogate models.

**2) Load surrogate model.** Load surrogate model that will be used to craft the adversarial attacks in the black-box setting.

**3) Craft adversarial attack.** Define attack configuration and craft the attack using the surrogate model.

**4) Load target model.** Load target model that will be attacked in the black-box setting. 

**5) Attack target model.** Obtain predictions in clean and adversarial setting. Visualization of original image, adversarial image, and corresponding adversarial noise.

The file makes use of the folder structure and files of the **data repository** that can be downloaded [here](http://doi.org/10.5281/zenodo.4792375). In that data repository:

The **metadata** folder contains:
- CSV files with the filenames of the images included in each additional data partition used in the paper:
    - Development sets: *d1*, *d110* (equivalent to *d1/10*), *d2*, and *d22* (equivalent to *d2/2*), and the corresponding filenames used for training (*t1*, **t110*, *t2*, *t22*) and validation (*v1*, *v110*, *v2*, *v22*) in each development set.

The **data** folder contains 2 example images (one with metastatic tissue present and one without) from the [PatchCamelyon dataset](https://github.com/basveeling/pcam) included in the test set from the publication. 

The **models** folder contains all the models trained for the publication that served as target or surrogate models in the black-box adversarial settings.

