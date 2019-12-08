import os
import pprint

import numpy as np
import pandas as pd
from paws.plugins.BayesianDesigner import BayesianDesigner

bd = BayesianDesigner(
    strategy='MPI',                                 # ignore
    strategic_params={'exploration_incentive':0.},  # ignore
    noise_sd=0.1,                                   # *** this can be tuned
    x_domain = dict(                                # ignore
        T_set=[220.,310.],
        flowrate=[20.,100.],
        oleylamine_fraction=[0.,0.3],
        TOP_fraction=[0.,0.3],
        ODE_extra_fraction=[0.,0.3]
        ),
    constraints = {'r0_sphere':10.},                # ignore
    range_constraints = dict(                       # ignore
        sigma_sphere=[0.,0.1],
        I0_fraction_sphere=[0.9,1.],
        I0_sphere=[1.E1,1.E6]
        ),
    categorical_constraints = dict(                 # ignore
        dilute_sphere_flag=1,
        disordered_flag=0,
        crystalline_flag=0
        ),
    covariance_kernel = 'sq_exp',                   # *** choose 'sq_exp' or 'inv_exp'
    covariance_kernel_params = {'width':1.},        # *** this can be tuned
    MC_max_iter = 1000.,                            # ignore
    MC_alpha = 1.,                                  # ignore
    verbose = True,                                 # ignore
    log_file = None                                 # ignore
    )
bd.start()

# specify training data csv file:
# change this to train models on different datasets
root_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(root_dir,'dataset_full.csv')

# read training data, train bd
df = pd.read_csv(csv_path)
bd.set_data(df)

# specify recipes:
# each recipe must define a value for each key in bd's x_domain input
recipes = [\
    dict(T_set=280.,flowrate=40.,oleylamine_fraction=0.1,TOP_fraction=0.1,ODE_extra_fraction=0.1),\
    dict(T_set=280.,flowrate=80.,oleylamine_fraction=0.1,TOP_fraction=0.1,ODE_extra_fraction=0.1),\
    ]

# get proper ordering directly from bd training dataframe
x_keys = list(bd.xs_df.columns)

# run predictions for all recipes
for rcp in recipes:
    # make a single-row array out of the recipe
    x_rcp = np.array([rcp[k] for k in x_keys]).reshape(1,-1)
    # standardize the array
    xs_rcp = bd.x_scaler.transform(x_rcp)
    # pull the first row out of the array and run the predictions
    preds,gp_preds,gp_scores = bd.predict_outputs(xs_rcp[0])
    # print things
    print('recipe:')
    pprint.pprint(rcp)
    print('predictions:')
    pprint.pprint(preds)   




