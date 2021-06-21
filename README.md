# Adversarial Attack Vulnerability of Medical Image Analysis Systems: Unexplored Factors

This is the code used in experiments presented in manuscript:

Bortsova, G., González-Gonzalo, C., Wetstein, S. C., Dubost, F., Katramados, I., Hogeweg, L., Liefers, B., van Ginneken, B., Pluim, J. P. W., Veta, M., Sánchez, C. I., & de Bruijne, M. (2021). **Adversarial Attack Vulnerability of Medical Image Analysis Systems: Unexplored Factors**. Medical Image Analysis, 102141. [https://doi.org/https://doi.org/10.1016/j.media.2021.102141](https://doi.org/https://doi.org/10.1016/j.media.2021.102141)

## Abstract

Adversarial attacks are considered a potentially serious security threat for machine learning systems. Medical image analysis (MedIA) systems have recently been argued to be vulnerable to adversarial attacks due to strong financial incentives and the associated technological infrastructure.

In this paper, we study previously unexplored factors affecting adversarial attack vulnerability of deep learning MedIA systems in three medical domains: ophthalmology, radiology, and pathology. We focus on adversarial black-box settings, in which the attacker does not have full access to the target model and usually uses another model, commonly referred to as surrogate model, to craft adversarial examples that are then transferred to the target model. We consider this to be the most realistic scenario for MedIA systems.
Firstly, we study the effect of weight initialization (pre-training on ImageNet or random initialization) on the transferability of adversarial attacks from the surrogate model to the target model, i.e., how effective attacks crafted using the surrogate model are on the target model. Secondly, we study the influence of differences in development (training and validation) data between target and surrogate models. 
We further study the interaction of weight initialization and data differences with differences in model architecture. All experiments were done with a perturbation degree tuned to ensure maximal transferability at minimal visual perceptibility of the attacks. 

Our experiments show that pre-training may dramatically increase the transferability of adversarial examples, even when the target and surrogate's architectures are different: the larger the performance gain using pre-training, the larger the transferability.
Differences in the development data between target and surrogate models considerably decrease the performance of the attack; this decrease is further amplified by difference in the model architecture.
We believe these factors should be considered when developing security-critical MedIA systems planned to be deployed in clinical practice. We recommend avoiding using only standard components, such as pre-trained architectures and publicly available datasets, as well as disclosure of design specifications, in addition to using adversarial defense methods. When evaluating the vulnerability of MedIA systems to adversarial attacks, various attack scenarios and target-surrogate differences should be simulated to achieve realistic robustness estimates.

## Details

"Ophthalmology", "Radiology", and "Pathology" folders contain code
implementing adversarial attacks (using one-step and interative FGSM attacks)
on networks trained on [Kaggle Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection/),
[ChestX-Ray14](https://www.kaggle.com/nih-chest-xrays/data), and
[PatchCamelyon (PCam)](https://github.com/basveeling/pcam) datasets, respectively.
The experiments on the three datasets were performed and the respective code was prepared
by Cristina Gonzalez-Gonzalo, Gerda Bortsova, and Suzanne C. Wetstein, respectively.

All models used in experiments in the manuscript can be downloaded [here](http://doi.org/10.5281/zenodo.4792375).