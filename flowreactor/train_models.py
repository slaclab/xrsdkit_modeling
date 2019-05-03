import os

import xrsdkit.tools.devtools as xrsdev

root_dir = os.path.abspath(os.path.dirname(__file__)) 
dataset_path = os.path.join(root_dir,'dataset')
output_path = os.path.join(root_dir,'models')
config_path = os.path.join(root_dir,'model_config.yml')

xrsdev.train_on_local_dataset(dataset_path,output_path,config_path)

