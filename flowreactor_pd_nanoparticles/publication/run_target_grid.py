import os

import numpy as np
import pandas as pd
import yaml

from paws.pawstools import primitives
from xrsdkit.tools import ymltools as xrsdyml
from paws.plugins.Timer import Timer
from paws.plugins.FlowReactor import FlowReactor
from paws.plugins.BayesianDesigner import BayesianDesigner
from paws.plugins.PyFAIIntegrator import PyFAIIntegrator
from paws.plugins.SSHClient import SSHClient
from paws.plugins.SpecInfoClient import SpecInfoClient
from paws.workflows.PATTERN_PROCESSING_1D.DezingerBatch import DezingerBatch
from paws.workflows.IMAGE_INTEGRATION.IntegrateBatch import IntegrateBatch
from paws.workflows.FLOW_REACTOR.SSRL_BEAMLINE_1_5.RunRecipeTakeImages import RunRecipeTakeImages
from paws.operations.ARRAYS.ArrayYMean import ArrayYMean
from paws.operations.BACKGROUND.BgSubtract import BgSubtract
from paws.operations.PATTERN_PROCESSING_1D.XrsdkitProcess1d import XrsdkitProcess1d
from tools import unpack_properties

expt_id = 'R1_20190424'
modeling_dataset_file = 'dataset_R0.csv'
root_dir = os.path.join('D:\\','20190424_flowreactor')
output_dir = os.path.join(root_dir,expt_id)
dataset_path = os.path.join(root_dir,modeling_dataset_file)
new_dataset_path = os.path.join(output_dir,'dataset_new.csv')
if not os.path.exists(output_dir): os.mkdir(output_dir)
plugin_log_dir = os.path.join(output_dir,'log_files')
if not os.path.exists(plugin_log_dir): os.mkdir(plugin_log_dir)
header_dir = os.path.join(output_dir,'headers')
if not os.path.exists(header_dir): os.mkdir(header_dir)
image_dir = os.path.join(output_dir,'images')
if not os.path.exists(image_dir): os.mkdir(image_dir)
data_dir = os.path.join(output_dir,'data')
if not os.path.exists(data_dir): os.mkdir(data_dir)
bg_dir = os.path.join(output_dir,'bg')
if not os.path.exists(bg_dir): os.mkdir(bg_dir)
bg_header_dir = os.path.join(output_dir,'bg','headers')
if not os.path.exists(bg_header_dir): os.mkdir(bg_header_dir)
bg_image_dir = os.path.join(output_dir,'bg','images')
if not os.path.exists(bg_image_dir): os.mkdir(bg_image_dir)
bg_data_dir = os.path.join(output_dir,'bg','data')
if not os.path.exists(bg_data_dir): os.mkdir(bg_data_dir)

# RUN SETTINGS
reactor_volume = 400.
reactor_flush_factor = 2.5
n_exposures = 5
MC_max_iter = 40000
MC_alpha = 0.05
bad_flow_tol = 200
cov_kernel_width = 1. 
target_grid = np.arange(10.,50.,4.)

# DBG SETTINGS
#reactor_volume = 1.
#n_exposures = 2
#MC_max_iter = 1000

exposure_time = 10.
polz_factor = 1. 
src_wl = 0.799898

# TODO: test the timeout functionality
timer = Timer(
    dt = 2.,
    t_max = 60.*60.*36,
    #verbose = True
    )
timer.start()

flow_reactor = FlowReactor(
    timer = timer,
    ppumps_setup = dict(
        ODE=dict(
            device='COM3',
            flowrate_table=np.loadtxt(os.path.join(root_dir,'ODE_high_flow_table.dat')),
            flowrate_sensitivity=10.,
            bad_flow_tol=bad_flow_tol,
            volume_limit=90000
            ),
        Pd_TOP=dict(
            device='COM4',
            flowrate_table=np.loadtxt(os.path.join(root_dir,'ODE_high_flow_table.dat')),
            flowrate_sensitivity=10.,
            bad_flow_tol=bad_flow_tol, 
            volume_limit=40000
            ),
        TOP=dict(
            device='COM5',
            flowrate_table=np.loadtxt(os.path.join(root_dir,'TOP_flow_table.dat')),
            flowrate_sensitivity=1.,
            bad_flow_tol=bad_flow_tol, 
            volume_limit=17000
            ),
        oleylamine=dict(
            device='COM6',
            flowrate_table=np.loadtxt(os.path.join(root_dir,'oleylamine_flow_table.dat')),
            flowrate_sensitivity=1.,
            bad_flow_tol=bad_flow_tol,
            volume_limit=17000
            ),
        ODE_extra=dict(
            device='COM7',
            flowrate_table=np.loadtxt(os.path.join(root_dir,'ODE_low_flow_table.dat')),
            flowrate_sensitivity=1.,
            bad_flow_tol=bad_flow_tol,
            volume_limit=17000
            )
        ),
    cryocon_setup = dict(
        host = '192.0.2.21',
        port = 5000,
        channels = {'A':3}
        ),
    verbose=True,
    log_file = os.path.join(plugin_log_dir,'flow_reactor.log')
    )
flow_reactor.start()

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
    covariance_kernel_params = {'width':cov_kernel_width},
    MC_max_iter = MC_max_iter,
    MC_alpha = MC_alpha,
    verbose = True,
    log_file = os.path.join(plugin_log_dir,'flow_designer.log')
    )
flow_designer.start()
dataset = pd.read_csv(dataset_path,index_col=0) 
flow_designer.set_data(dataset)

integrator = PyFAIIntegrator(
    calib_file = os.path.join(root_dir,'AgBe.nika'),
    q_min = 0.03,
    q_max = 0.6,
    verbose = True
    )
integrator.start()

mar_ssh_client = SSHClient(
    username = 'marccd',
    hostname = '192.0.2.30',
    port = 22,
    password = 'FeRuOs3121',
    verbose = True
    )
mar_ssh_client.start()

spec_infoclient = SpecInfoClient(
    host = '192.0.2.2',
    port = 2034,
    verbose=True
    )
spec_infoclient.start()

dz = DezingerBatch()
integ = IntegrateBatch()
Imean = ArrayYMean()
bgsub = BgSubtract()
run_and_image = RunRecipeTakeImages()
xrsdkit_process1d = XrsdkitProcess1d()

first_trialno = 3
final_trialno = len(target_grid)-1
target_grid = target_grid[first_trialno:]
trialnos = range(first_trialno,final_trialno+1)

import pdb; pdb.set_trace()

for trialno,r0_target in zip(trialnos,target_grid):

    # if first iteration, start optimizing first target
    if trialno == first_trialno:
        flow_designer.set_constraints(r0_sphere=r0_target)
        flow_designer.optimize_candidate()

    # get next available candidate (blocks if not yet available)
    cand_data = flow_designer.get_next_candidate()
    rcp = cand_data['candidate']

    # if not final iteration, start optimizing next candidate
    if not trialno == final_trialno:
        flow_designer.set_constraints(r0_sphere=target_grid[trialno+1])
        flow_designer.optimize_candidate()

    # parse recipe into flowreactor inputs
    flowreac_rxn_rcp = {'T_set':rcp['T_set'],'T_ramp':60.} 
    flowreac_bg_rcp = {'T_set':rcp['T_set'],'T_ramp':60.} 
    total_flowrate = rcp['flowrate']
    solvent_frac = 1.
    for pump_nm in ['TOP','oleylamine','ODE_extra']:
        reagent_frac = rcp['{}_fraction'.format(pump_nm)] 
        solvent_frac -= reagent_frac 
        flowreac_rxn_rcp['{}_flowrate'.format(pump_nm)] = reagent_frac*total_flowrate
        flowreac_bg_rcp['{}_flowrate'.format(pump_nm)] = reagent_frac*total_flowrate
    flowreac_bg_rcp['Pd_TOP_flowrate'] = 0.
    flowreac_bg_rcp['ODE_flowrate'] = solvent_frac * total_flowrate
    flowreac_rxn_rcp['Pd_TOP_flowrate'] = solvent_frac * total_flowrate
    flowreac_rxn_rcp['ODE_flowrate'] = 0.
    delay_time = reactor_flush_factor*reactor_volume*60./total_flowrate 

    # start a dict of header data
    rxn_id = expt_id+'_{}'.format(trialno)
    bg_rxn_id = expt_id+'_{}_bg'.format(trialno)
    tmstmp = flow_reactor.timer.get_epoch_time()
    hdr_data = dict(source_wavelength=src_wl,
            experiment_id=expt_id,
            exposure_time=exposure_time,
            time=tmstmp
            )
    hdr_data.update(rcp)

    # set background recipe, take images
    bg_outputs = run_and_image.run_with(
            flow_reactor=flow_reactor,
            spec_infoclient=spec_infoclient,
            ssh_client=mar_ssh_client,
            ssh_data_dir='/home/data',
            recipe=flowreac_bg_rcp,
            delay_time=delay_time,
            n_exposures=n_exposures,
            exposure_time=10.,
            header_data=hdr_data,
            reaction_id=bg_rxn_id,
            header_output_dir=bg_header_dir,
            image_output_dir=bg_image_dir
            )

    # set reaction recipe, take images
    rxn_outputs = run_and_image.run_with(
            flow_reactor=flow_reactor,
            spec_infoclient=spec_infoclient,
            ssh_client=mar_ssh_client,
            ssh_data_dir='/home/data',
            recipe=flowreac_rxn_rcp,
            delay_time=delay_time,
            n_exposures=n_exposures,
            exposure_time=10.,
            header_data=hdr_data,
            reaction_id=rxn_id,
            header_output_dir=header_dir,
            image_output_dir=image_dir
            )

    # integrate all images to 1d patterns
    bg_integ_outputs = integ.run_with(
            integrator=integrator,
            images=bg_outputs['images'],
            image_paths=bg_outputs['image_paths'],
            n_points=1000,
            polz_factor=1.,
            output_dir=bg_data_dir)
    rxn_integ_outputs = integ.run_with(
            integrator=integrator,
            images=rxn_outputs['images'],
            image_paths=rxn_outputs['image_paths'],
            n_points=1000,
            polz_factor=1.,
            output_dir=data_dir)

    # dezinger all patterns
    bg_dz_outputs = dz.run_with(
            q_I_arrays=bg_integ_outputs['data'],
            q_I_paths=bg_integ_outputs['data_paths'],
            output_dir=bg_data_dir
            )
    rxn_dz_outputs = dz.run_with(
            q_I_arrays=rxn_integ_outputs['data'],
            q_I_paths=rxn_integ_outputs['data_paths'],
            output_dir=data_dir
            )

    # average bg and reaction patterns
    bg_q_Imean = Imean.run_with(x_y_arrays=bg_dz_outputs['data'])['x_ymean']
    rxn_q_Imean = Imean.run_with(x_y_arrays=rxn_dz_outputs['data'])['x_ymean']
    bg_mean_fn = os.path.join(bg_data_dir,bg_rxn_id+'_dz.dat') 
    rxn_mean_fn = os.path.join(data_dir,rxn_id+'_dz.dat')
    np.savetxt(bg_mean_fn,bg_q_Imean,delimiter=' ',header='q (1/Angstrom), I (arb)')
    np.savetxt(rxn_mean_fn,rxn_q_Imean,delimiter=' ',header='q (1/Angstrom), I (arb)')

    # subtract bg from reaction
    q_I_bgsub = bgsub.run_with(q_I=rxn_q_Imean,q_I_bg=bg_q_Imean)['q_I_bgsub']
    bgsub_fn = os.path.join(data_dir,rxn_id+'_dz_bgsub.dat')
    np.savetxt(bgsub_fn,q_I_bgsub,delimiter=' ',header='q (1/Angstrom), I (arb)')

    # classify and fit with xrsdkit
    sample_metadata = dict(
        source_wavelength=src_wl, 
        experiment_id=expt_id, 
        sample_id=rxn_outputs['headers'][-1]['sample_id'],
        data_file=os.path.split(bgsub_fn)[1],
        time=tmstmp 
        )
    sys_opt = xrsdkit_process1d.run_with(
            q_I=q_I_bgsub,
            sample_metadata=sample_metadata,
            fit_args={'error_weighted':True,'logI_weighted':True,'q_range':[0.,float('inf')]}
            )['xrsd_system']

    # save candidate recipe and xrsd system results
    sys_fn = os.path.join(data_dir,rxn_id+'_xrsd_system.yml')
    xrsdyml.save_sys_to_yaml(sys_fn,sys_opt)
    candidate_fn = os.path.join(data_dir,rxn_id+'_candidate_data.yml')
    yaml.dump(primitives(cand_data),open(candidate_fn,'w'))

    # (update the dataset)
    new_sample = {'experiment_id':expt_id,'sample_id':sample_metadata['sample_id']}
    new_sample.update(rcp)
    new_sample.update(unpack_properties(sys_opt)) 
    dataset = dataset.append(new_sample,ignore_index=True)
    dataset.to_csv(new_dataset_path)

    # (retrain the designer)
    # ...

flow_reactor.stop()
timer.stop()
    
