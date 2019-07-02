import os

from xrsdkit.tools import ymltools as xrsdyml 
from paws.workflows.SSRL_BEAMLINE_1_5.LEGACY.ReadTimeSeries import ReadTimeSeries

root_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(root_dir,'dataset')

#src_wl = 0.799898
#expt_id = 'R1_201611'
#expt_id = 'R2_201611'
#expt_id = 'R3_201611'
#expt_id = 'R4_201611'
#expt_id = 'R5_201611'
#expt_id = 'R6_201611'
#expt_id = 'R7_201611'
#expt_id = 'RxnA_201602'
#expt_id = 'RxnB_201602'
#expt_id = 'RxnC_201602'
#expt_id = 'RxnD_201602'
#expt_id = 'RxnE_201602'
#expt_id = 'RxnF_201602'
#expt_id = 'RxnG_201602'
#expt_id = 'RxnH_201602'
#expt_id = 'RxnK_201602'

src_wl = 0.826562
#expt_id = 'R12_201607'
expt_id = 'R13_201607'

expt_dir = os.path.join(root_dir,expt_id)
hdr_dir = os.path.join(expt_dir,'headers')

reader = ReadTimeSeries()
timeseries_outputs = reader.run_with(
    header_dir = hdr_dir,
    header_regex = '*.txt',
    q_I_dir = expt_dir, 
    q_I_suffix = '_dz_bgsub',
    q_I_ext = '.dat',
    system_dir = expt_dir, 
    system_suffix = '_dz_bgsub'
    )

#import pdb; pdb.set_trace()

for sysf,datf,sys,tmstmp in zip(
            timeseries_outputs['system_files'],
            timeseries_outputs['q_I_files'],
            timeseries_outputs['system'],
            timeseries_outputs['time']):
    if os.path.exists(sysf):
        if not sys.sample_metadata['experiment_id']:
            print('found missing experiment_id, assigning {}'.format(expt_id))
            sys.sample_metadata['experiment_id'] = expt_id
        if not sys.sample_metadata['source_wavelength'] == src_wl:
            print('found missing/incorrect source_wavelength, assigning {}'.format(src_wl))
            sys.sample_metadata['source_wavelength'] = src_wl 
        datfnm = os.path.split(datf)[1]
        if not sys.sample_metadata['data_file'] == datfnm:
            print('found missing/incorrect data file, assigning {}'.format(datfnm))
            sys.sample_metadata['data_file'] = datfnm 
        if not sys.sample_metadata['sample_id']:
            sampid = expt_id+'_'+str(int(tmstmp))
            print('found missing sample_id, assigning {}'.format(sampid))
            sys.sample_metadata['sample_id'] = sampid
        if not sys.fit_report['good_fit']:
            print('system definition file {} indicates bad fit'.format(sysf))
        xrsdyml.save_sys_to_yaml(sysf,sys)
    else:
        print('missing system definition file: {}'.format(sysf))


