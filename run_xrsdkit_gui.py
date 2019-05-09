from collections import OrderedDict
import copy
import glob
import os

import yaml

from xrsdkit.visualization.gui import run_gui

run_gui()

#root_dir = os.path.abspath(os.path.dirname(__file__)) 
#dataset_path = os.path.join(root_dir,'flowreactor','dataset')
#expt_id = 'R2_20190424'
#expt_dir = os.path.join(dataset_path,expt_id)

#yml_files = glob.glob(os.path.join(expt_dir,'*.yml'))
#sys_data = [yaml.load(open(fp,'r')) for fp in yml_files]
#data_files = [os.path.join(expt_dir,sd['sample_metadata']['data_file']) for sd in sys_data]

#data_map = dict.fromkeys(data_files)
#for df,yf in zip(data_files,yml_files):
#    data_map[df] = yf



