import os
import glob

import pandas as pd
from paws.workflows.SSRL_BEAMLINE_1_5.ReadTimeSeries import ReadTimeSeries

def unpack_properties(sys):
    # unpack and label the properties of interest
    # (classification flags and physical parameters)
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

root_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(root_dir,'dataset')
dataset_file = os.path.join(dataset_dir,'dataset.csv')
expt_dirs = glob.glob(os.path.join(dataset_dir,'*'))

recipe_keys = ['flowrate','T_set','TOP_fraction','oleylamine_fraction','ODE_extra_fraction']
property_keys = ['r0_sphere','sigma_sphere','I0_sphere','I0_fraction_sphere','disordered_flag','crystalline_flag','dilute_sphere_flag']

ds = pd.DataFrame(columns=['sample_id','experiment_id']+recipe_keys+property_keys)

reader = ReadTimeSeries()

for expt_dir in expt_dirs:
    if os.path.isdir(expt_dir):
        hdr_dir = os.path.join(expt_dir,'headers')
        expt_data = reader.run_with(
                header_dir = hdr_dir,
                header_regex = '*.yml',
                system_dir = expt_dir,
                system_suffix = '_xrsd_system'
                )
        for hdr_data,xrsdsys in zip(expt_data['header_data'],expt_data['system']):
            if xrsdsys.fit_report['good_fit']:
                new_sample = {} 
                new_sample['sample_id'] = xrsdsys.sample_metadata['sample_id']
                new_sample['experiment_id'] = xrsdsys.sample_metadata['experiment_id']
                for rcpk in recipe_keys:
                    new_sample[rcpk] = hdr_data[rcpk]
                xrsd_props = unpack_properties(xrsdsys)
                for propk in property_keys:
                    new_sample[propk] = xrsd_props[propk]
                ds = ds.append(new_sample,ignore_index=True)

ds.to_csv(dataset_file,index=False)

