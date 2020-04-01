# Adverserial Regularizers in Inverse Problems
Code for the paper [Adversarial Regularizer in Inverse Problems](https://arxiv.org/abs/1805.11572).

Inverse Problems are traditionally solved using purely model-based methods, as in variational regularization methods. 
We propose a new framework for applying data-driven approaches to inverse problems, using a neural network as
regularization functional. 

# Method

The network is trained as a critic as in [WGANs](https://arxiv.org/abs/1701.07875), learning to discriminate between the
distribution of ground truth images and the distribution of unregularized reconstruction.

Once trained, we use the learned regularization functional to solve inverse problems by minizing the associated variational functional. In the context of computed tomography we employ early stopping to obtain the best results.

# Results
Reconstructions obtained on the LIDC/IDRI dataset
<img src="http://www.damtp.cam.ac.uk/user/sl767/picture_ar.png" 
width="900" title="Results_LIDC">

# Getting Started

Download the [LIDC_IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) and/or the
[BSDS500](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz) dataset and split the raw 
data into two folders named Training_Data and Evaluation_Data inside your DATA_PATH folder.

The package requirements are numpy, tensorflow, scipy, scikit-image, matplotlib, 
[pydicom](https://pydicom.github.io/pydicom/stable/getting_started.html), 
[astra-toolbox](https://www.astra-toolbox.com/docs/install.html), and [odl](https://github.com/odlgroup/odl). 
All requirements can be installed directly into the conda environment advReg via
```bash
$ conda env create -f advReg.yml
$ source activate advReg
```

The tensorboard loggings can be found in SAVES_PATH.

# Customization

To add custom data sets, forward operators for different inverse problems or network architectures, write your custom 
implementation of the corresponding abstract classes in ClassFiles. Finally, set the
get_Data_pip, get_model or get_network in your experiments accordingly.
