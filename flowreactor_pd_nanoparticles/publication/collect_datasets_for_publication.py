import os
import glob

import numpy as np
import pandas as pd
from paws.workflows.SSRL_BEAMLINE_1_5.ReadTimeSeries import ReadTimeSeries
from paws.plugins.BayesianDesigner import BayesianDesigner

# subroutine for unpacking a dict of interesting properties from an xrsdkit system
def unpack_properties(sys):
    dilute_sphere_flag = False
    disordered_flag = False
    crystalline_flag = False
    r0_sphere = None 
    sigma_sphere = None 
    I0_sphere = None 
    I0_fraction_sphere = None 
    I0_tot = 0.
    for popnm,pop in sys.populations.items():    
        I0_tot += pop.parameters['I0']['value']
        if pop.structure == 'disordered':
            disordered_flag = True
        if pop.structure == 'crystalline':
            crystalline_flag = True
        if pop.structure == 'diffuse':
            if pop.form == 'spherical':
                if pop.settings['distribution'] == 'r_normal':
                    dilute_sphere_flag = True
                    r0_sphere = pop.parameters['r']['value']
                    sigma_sphere = pop.parameters['sigma']['value']
                    I0_sphere = pop.parameters['I0']['value']
    if I0_sphere:
        I0_fraction_sphere = I0_sphere/I0_tot
    props = {} 
    props.update(
        dilute_sphere_flag = dilute_sphere_flag,
        disordered_flag = disordered_flag,
        crystalline_flag = crystalline_flag,
        r0_sphere = r0_sphere,
        sigma_sphere = sigma_sphere,
        I0_sphere = I0_sphere,
        I0_fraction_sphere = I0_fraction_sphere
        )
    return props

# set up and start the design tool as-used in the experiment
flow_designer = BayesianDesigner(
    strategy = 'MPI', 
    strategic_params = {'exploration_incentive':0.},
    noise_sd = 0.1,
    x_domain = dict(
        T_set=[220.,300.],
        flowrate=[20.,120.],
        oleylamine_fraction=[0.,0.3],
        TOP_fraction=[0.,0.3],
        ODE_extra_fraction=[0.,0.3]
        ),
    targets = {},
    constraints = {'r0_sphere':10.},
    range_constraints = dict( 
        sigma_sphere=[None,0.2],
        I0_fraction_sphere=[0.9,None],
        I0_sphere=[1.E1,None]
        ),
    categorical_constraints = dict(
        dilute_sphere_flag=1,
        disordered_flag=0,
        crystalline_flag=0
        ),
    covariance_kernel = 'sq_exp',
    covariance_kernel_params = {'width':1.},
    MC_max_iter = 40000,
    MC_alpha = 0.05,
    verbose = True,
    log_file = None
    )
flow_designer.start()

# set up the dataframes to hold the dataset tables
recipe_keys = ['flowrate','T_set','TOP_fraction','oleylamine_fraction','ODE_extra_fraction']
property_keys = ['r0_sphere','sigma_sphere','I0_sphere','I0_fraction_sphere','disordered_flag','crystalline_flag','dilute_sphere_flag']
pred_keys = [k+'_pred' for k in property_keys] 
pred_var_keys = [k+'_pred_var' for k in property_keys]
ds_init = pd.DataFrame(columns=['sample_id','experiment_id']+recipe_keys+property_keys)
ds_R0 = pd.DataFrame(columns=['sample_id','experiment_id']+recipe_keys+property_keys+pred_keys+pred_var_keys)
ds_R1 = pd.DataFrame(columns=['sample_id','experiment_id']+recipe_keys+property_keys+pred_keys+pred_var_keys)
ds_R2 = pd.DataFrame(columns=['sample_id','experiment_id']+recipe_keys+property_keys+pred_keys+pred_var_keys)


# use the timeseries reader to build the initial grid-search dataset
reader = ReadTimeSeries()
system_dir = os.path.join('..','dataset','R0_201811')
header_dir = os.path.join(system_dir,'headers')
data_init = reader.run_with(
            header_dir = header_dir,
            header_regex = '*.yml',
            system_dir = system_dir,
            system_suffix = '_dz_bgsub')
for hdr_data,xrsdsys in zip(data_init['header_data'],data_init['system']):
    new_sample = {} 
    new_sample['sample_id'] = xrsdsys.sample_metadata['sample_id']
    new_sample['experiment_id'] = xrsdsys.sample_metadata['experiment_id']
    for rcpk in recipe_keys:
        new_sample[rcpk] = hdr_data[rcpk]
    if xrsdsys.fit_report['good_fit']:
        xrsd_props = unpack_properties(xrsdsys)
        for propk in property_keys:
            new_sample[propk] = xrsd_props[propk]
    ds_init = ds_init.append(new_sample,ignore_index=True)
ds_init.to_csv('dataset_init.csv',index=False)

# train the design tool on the grid-search dataset
flow_designer.set_data(ds_init)

# get recipe key ordering directly from training dataframe
x_keys = list(flow_designer.xs_df.columns)

# use timeseries reader and design tool to build the R0 dataset+predictions
system_dir = os.path.join('..','dataset','R0_20190424')
header_dir = os.path.join(system_dir,'headers')
data_R0 = reader.run_with(
            header_dir = header_dir,
            header_regex = '*.yml',
            system_dir = system_dir,
            system_suffix = '_dz_bgsub')
for hdr_data,xrsdsys in zip(data_R0['header_data'],data_R0['system']):
    new_sample = {}
    new_sample['sample_id'] = xrsdsys.sample_metadata['sample_id']
    new_sample['experiment_id'] = xrsdsys.sample_metadata['experiment_id']
    for rcpk in recipe_keys:
        new_sample[rcpk] = hdr_data[rcpk]
    # fill out predictions
    x_rcp = np.array([hdr_data[k] for k in x_keys]).reshape(1,-1)
    xs_rcp = flow_designer.x_scaler.transform(x_rcp).ravel()
    preds,gp_preds,gp_scores = flow_designer.predict_outputs(xs_rcp)
    for k in preds.keys():
        new_sample[k+'_pred'] = preds[k][0]
        new_sample[k+'_pred_var'] = preds[k][1]
    if xrsdsys.fit_report['good_fit']:
        xrsd_props = unpack_properties(xrsdsys)
        for propk in property_keys:
            new_sample[propk] = xrsd_props[propk]
    ds_R0 = ds_R0.append(new_sample,ignore_index=True)
ds_R0.to_csv('dataset_R0.csv',index=False)




# retrain the design tool on the R0 dataset
flow_designer.set_data(ds_R0)

# get recipe key ordering directly from training dataframe
x_keys = list(flow_designer.xs_df.columns)

# use timeseries reader and design tool to build the R1 dataset+predictions
system_dir = os.path.join('..','dataset','R1_20190424')
header_dir = os.path.join(system_dir,'headers')
data_R1 = reader.run_with(
            header_dir = header_dir,
            header_regex = '*.yml',
            system_dir = system_dir,
            system_suffix = '_dz_bgsub')
for hdr_data,xrsdsys in zip(data_R1['header_data'],data_R1['system']):
    new_sample = {}
    new_sample['sample_id'] = xrsdsys.sample_metadata['sample_id']
    new_sample['experiment_id'] = xrsdsys.sample_metadata['experiment_id']
    for rcpk in recipe_keys:
        new_sample[rcpk] = hdr_data[rcpk]
    # fill out predictions
    x_rcp = np.array([hdr_data[k] for k in x_keys]).reshape(1,-1)
    xs_rcp = flow_designer.x_scaler.transform(x_rcp).ravel()
    preds,gp_preds,gp_scores = flow_designer.predict_outputs(xs_rcp)
    for k in preds.keys():
        new_sample[k+'_pred'] = preds[k][0]
        new_sample[k+'_pred_var'] = preds[k][1]
    if xrsdsys.fit_report['good_fit']:
        xrsd_props = unpack_properties(xrsdsys)
        for propk in property_keys:
            new_sample[propk] = xrsd_props[propk]
    ds_R1 = ds_R1.append(new_sample,ignore_index=True)
ds_R1.to_csv('dataset_R1.csv',index=False)




# retrain the design tool on the R1 dataset
flow_designer.set_data(ds_R1)

# get recipe key ordering directly from training dataframe
x_keys = list(flow_designer.xs_df.columns)

# use timeseries reader and design tool to build the R2 dataset+predictions
system_dir = os.path.join('..','dataset','R2_20190424')
header_dir = os.path.join(system_dir,'headers')
data_R2 = reader.run_with(
            header_dir = header_dir,
            header_regex = '*.yml',
            system_dir = system_dir,
            system_suffix = '_dz_bgsub')
for hdr_data,xrsdsys in zip(data_R2['header_data'],data_R2['system']):
    new_sample = {}
    new_sample['sample_id'] = xrsdsys.sample_metadata['sample_id']
    new_sample['experiment_id'] = xrsdsys.sample_metadata['experiment_id']
    for rcpk in recipe_keys:
        new_sample[rcpk] = hdr_data[rcpk]
    # fill out predictions
    x_rcp = np.array([hdr_data[k] for k in x_keys]).reshape(1,-1)
    xs_rcp = flow_designer.x_scaler.transform(x_rcp).ravel()
    preds,gp_preds,gp_scores = flow_designer.predict_outputs(xs_rcp)
    for k in preds.keys():
        new_sample[k+'_pred'] = preds[k][0]
        new_sample[k+'_pred_var'] = preds[k][1]
    if xrsdsys.fit_report['good_fit']:
        xrsd_props = unpack_properties(xrsdsys)
        for propk in property_keys:
            new_sample[propk] = xrsd_props[propk]
    ds_R2 = ds_R2.append(new_sample,ignore_index=True)
ds_R2.to_csv('dataset_R2.csv',index=False)



