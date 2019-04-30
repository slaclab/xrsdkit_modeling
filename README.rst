Flow Reactor Modeling
---------------------

This repository is for curating scattering patterns and xrsdkit analysis results
for colloidal nanoparticle samples from the flow reactor at beam line 1-5.

Usage instructions are below.
For a Python notebook that walks through this process,
see the usage_notebook.ipynb.


Disclaimer
----------

The contents of this repository can be used
to train the models of an xrsdkit installation
to perform automated analysis of scattering patterns.
The models trained by these data will be most effective
in analyzing data that are similar to the training set samples.
If an xrsdkit installation is trained here
and then used for predictions on unrelated samples,
strange results should be expected.


Usage
-----

It is expected that the user has installed a Python interpreter (preferably version 3.5 or greater),
and that the user has installed xrsdkit for that interpreter.
It is suggested to use virtual environments to avoid conflicts between Pythons,
especially if the user is using different xrsdkit model sets for different applications.


Step 1: Download this Repository
================================

Download this repository through the GitHub interface,
or clone it with git clone. 

    git clone https://github.com/slaclab/flowreactor_modeling.git 

Activate your virtual environment (presume that you created one and named it flowreactor),
and move into the directory you just downloaded.

    workon flowreactor
    cd flowreactor_modeling

If you plan to use the models as-distributed in this repository,
you can skip ahead to the last step, "Load Models".


Step 2a (optional): Update the dataset
======================================

If you are not planning to use the default dataset 
that is included in this repository,
you should now augment or replace the dataset directory 
with your xrsdkit-processed results (yml files grouped by experiment).
The dataset directory contains one or more subdirectories,
where each subdirectory consolidates xrsdkit results for one experiment. 
One experiment consists of one or more samples,
and one sample is represented by exactly one yml file.


Step 2b (optional): Refresh the model configuration 
===================================================

The model configuration file defines what kind of model is trained
for each estimator in the xrsdkit model set,
as well as any choices that can be made in training the model
(e.g. selection of metrics to optimize 
during automated feature selection and hyperparameter tuning).
To start with a fresh model configuration (to use xrsdkit defaults),
simply remove the model configuration file.

    rm model_config.yml

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


Step 3b (optional): Update the Configuration
============================================

The training process will generate a model configuration file,
which will be saved in this directory.
Edit the file with any text editor to select 
whatever model types and training metrics
are appropriate for the project at hand.
If you prefer to edit the configuration programmatically,
the configuration file can be loaded into a Python dictionary,
edited, and re-saved.

After editing the configuration, re-run the training 
to re-train the models in the new configuration.


Step 4: Load Models
===================

The trained model parameters are saved in a tree of files,
rooted at the modeling_data directory.
Each model is defined by a text file that describes its training results,
and a pickle file that can be used to rebuild the model itself. 
You can load these models into an xrsdkit module at runtime.

    import xrsdkit 
    xrsdkit.load_models('path/to/modeling/data/dir')

After executing these lines, the models defined in modeling_data
will be used for the duration of the runtime,
whenever xrsdkit.predict is called.

    import numpy as np
    q_I = np.loadtxt('path/to/1d/scattering/pattern.dat')
    feats = xrsdkit.profile_1d_pattern(q_I[:,0],q_I[:,1])
    preds = xrsdkit.predict(feats)
    xsys = xrsdkit.build_system_from_predictions(preds) 
    xfig = xrsdkit.plot_xrsd_fit(xsys,q_I[:,0],q_I[:,1])

