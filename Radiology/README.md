# Adversarial attacks on ChestX-Ray14 data

## Software prerequisites

The code requires TensorFlow 1.14 and other python packages listed in [requirements.txt](https://github.com/Gerda92/adversarial_transfer_factors/blob/2de1737c389049aec81bc81fb19b3502a087653d/Radiology/requirements.txt).
Using an earlier version of h5py (2.10.0) is important for building models.
The code was written and tested in Python 3.6 (the latest version of Python, 3.8, is not compatible with TF 1.14).

## Other prerequisites

The ChestX-Ray14 dataset can be downloaded [here](https://www.kaggle.com/nih-chest-xrays/data).
The code requires "Data_Entry_2017.csv" file and **the images downsampled to a 256 x 256 resolution**.
The former and a folder contaning the latter should be in the same directory,
which should be specified as `data_path` attribute of the parameters object
(see [base_params.yml](https://github.com/Gerda92/adversarial_transfer_factors/blob/2de1737c389049aec81bc81fb19b3502a087653d/Radiology/base_params.yml) and [params.yml](https://github.com/Gerda92/adversarial_transfer_factors/blob/2de1737c389049aec81bc81fb19b3502a087653d/Radiology/params.yml)).
The name of the folder with images should be specified as `images_folder` attribute of the parameters object.

Trained models can be downloaded [here](link). The path to them should be specified
as `model_path` attribute of the parameters object.

Attribute `results_path` should specify a name of an existing folder to save the experimental results to.

## Run a small experiment

[example_experiment.py](https://github.com/Gerda92/adversarial_transfer_factors/blob/2de1737c389049aec81bc81fb19b3502a087653d/Radiology/example_experiment.py) executes an attack specified in [params.yml](https://github.com/Gerda92/adversarial_transfer_factors/blob/2de1737c389049aec81bc81fb19b3502a087653d/Radiology/params.yml)
on a sample of test set images and plots the results.

## Run experiments presented in the manuscript

`compute_table_{table_number}.py` files execute experiments reported in Tables 2, 3, and 4 of the manuscript
and save the results as .csv tables.

## File structure
    .
    ├── adv_attack_func.py          # Functions for computing adversarial perturbations.
    ├── base_params.yml             # Basis parameters to execute experiments in compute_table_{table_number}.py files.
    ├── build_model.py              # A function for building target and surrogate models.
    ├── generate_final_tables.py    # Use outcomes of compute_table_{table_number}.py files to generate tables as shown in the supplement and the manuscript.
    ├── compute_attacks.py          # A function for computing attacks and testing them
    ├── compute_table_four.py       # Run experiments reported in Table 5 of the manuscript 
    ├── compute_table_one.py        # Run experiments reported in Table 2 of the manuscript 
    ├── compute_table_three.py      # Run experiments reported in Table 4 of the manuscript 
    ├── compute_table_two.py        # Run experiments reported in Table 3 of the manuscript 
    ├── example_experiment.py       # A script performing a small experiment and visualizing the results.
    ├── helpers.py                  # Helper functions.
    ├── improc.py                   # Functions for image loading and processing.
    ├── params.yml                  # An example of a parameters file.
    ├── README.md
    ├── requirements.txt            # Required Python packages
    └── splits                      # Training-validation-test splits used in the manuscript;
        ├── d0_filenames.yml            # all splits share the same test set.
        ├── d0_indices_data_entry.yml   # 'd0' corresponds to 'd1/10' in the manuscript;
        ├── d1_filenames.yml            # 'd3' corresponds to 'd2/2'.
        ├── d1_indices_data_entry.yml
        ├── d2_filenames.yml
        ├── d2_indices_data_entry.yml
        ├── d3_filenames.yml
        └── d3_indices_data_entry.yml



