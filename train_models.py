import os

import xrsdkit.tools.devtools as xrsdev

root_dir = os.path.abspath(os.path.dirname(__file__)) 
dataset_path1 = os.path.join(root_dir,'flowreactor_pd_nanoparticles','dataset')
dataset_path2 = os.path.join(root_dir,'batch_pd_nanoparticles','dataset')
output_path = os.path.join(root_dir,'models_all_data')
config_path = os.path.join(root_dir,'model_config_all_data.yml')

xrsdev.train_on_local_dataset([dataset_path1,dataset_path2],output_path,config_path)
