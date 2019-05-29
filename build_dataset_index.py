import os
from xrsdkit.tools.ymltools import read_local_dataset

this_dir = os.path.abspath(os.path.dirname(__file__))

for dsname in ['batch_pd_nanoparticles','flowreactor_pd_nanoparticles']:
    dataset_dir = os.path.join(this_dir,dsname,'dataset')
    idx_file_path = os.path.join(dataset_dir,'dataset_index.csv')
    df, idx_df = read_local_dataset(dataset_dir)
    idx_df.to_csv(idx_file_path)

