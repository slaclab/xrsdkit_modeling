xrsdkit modeling
----------------

This repository contains datasets from projects at SLAC/SSRL,
xrsdkit models trained on these datasets,
and notebooks for (re-)training and inspecting the models. 


Usage
-----

It is expected that the user has installed a Python interpreter 
(preferably version 3.5 or greater),
and that the user has installed xrsdkit for that interpreter.
Follow the installation instructions in the xrsdkit documentation, 
`here <https://xrsdkit.readthedocs.io/en/latest/>`_.

Usage is summarized in the step-by-step instructions below.
For examples and details, see the notebooks.

Step 1: Download this Repository
================================

Download this repository through the GitHub interface,
or clone it with git clone. ::

    $ git clone https://github.com/slaclab/flowreactor_modeling.git 

Optionally, activate your virtual environment 
(presume it is named "xrsdkit"),
and move into the repository root directory. ::

    $ workon xrsdkit 
    $ cd xrsdkit_modeling


Step 2a (optional): Add/update dataset
======================================

Add or augment existing datasets 
with xrsdkit-processed results (yml files).
The dataset directory contains one or more subdirectories,
where each subdirectory consolidates xrsdkit results for one experiment. 
One experiment consists of one or more samples,
and one sample is represented by exactly one yml file.


Step 2b (optional): Update the model configuration 
==================================================

The model configuration file defines what kind of model is trained
for each estimator in the xrsdkit model set,
as well as any choices that can be made in training the model
(e.g. feature selection, hyperparameter selction, and performance metrics to optimize).
To use xrsdkit defaults, do not provide a model configuration file.
If this file is left in place, it will be found during the next step,
and any applicable configurations will be carried over into the new models.


Step 3a: Train Models
=====================

Use python train_models.py to call on xrsdkit 
to train its models from data in the dataset directory.
This will save the trained model parameters and training metrics
in the modeling_data directory.
This overwrites the default models 
that are originally included in modeling_data.
This also overwrites the model configuration file model_config.yml.

    python train_models.py


Step 3b (optional): Update the model configuration (again)
==========================================================

The training process will generate a model configuration file,
which will be saved in the model output directory.
Edit the file with any text editor to select 
whatever model types and training metrics
are appropriate for the project at hand.

After editing the configuration, 
re-train the models in the new configuration.


Step 4: Load Models
===================

The trained model parameters are saved in a tree of files
in the model output directory.
Each model presents a text file that describes its training results,
and a pickle file that can be used to rebuild the model itself. 
You can load these models into xrsdkit at runtime. ::

    >>> import xrsdkit 
    >>> xrsdkit.load_models('path/to/modeling/dir')

After this, the models will be used whenever xrsdkit.predict is called. ::

    >>> import numpy as np
    >>> q_I = np.loadtxt('path/to/1d/scattering/pattern.dat')
    >>> feats = xrsdkit.profile_1d_pattern(q_I[:,0],q_I[:,1])
    >>> preds = xrsdkit.predict(feats)
    >>> xsys = xrsdkit.build_system_from_predictions(preds) 
    >>> xfig = xrsdkit.plot_xrsd_fit(xsys,q_I[:,0],q_I[:,1])

