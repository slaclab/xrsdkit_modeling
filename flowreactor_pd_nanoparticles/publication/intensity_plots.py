import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import yaml


R0_fpaths = [os.path.join('..','dataset','R0_20190424','R0_20190424_{}_dz_bgsub.yml'.format(idx)) for idx in range(9)]
results_R0 = [yaml.load(open(fpath,'r')) for fpath in R0_fpaths]
I_dilutesphere_R0 = np.zeros(9)
I_precursors_R0 = np.zeros(9)
I_disorderedsphere_R0 = np.zeros(9)

for idx,res in enumerate(results_R0):
    for k in res.keys():
        if not k in ['features','fit_report','noise','sample_metadata']:
            if res[k]['structure'] == 'diffuse' and res[k]['form'] == 'spherical':
                I_dilutesphere_R0[idx] = res[k]['parameters']['I0']['value']
            elif res[k]['structure'] == 'diffuse' and res[k]['form'] == 'guinier_porod':
                I_precursors_R0[idx] = res[k]['parameters']['I0']['value']
            elif res[k]['structure'] == 'disordered' and res[k]['form'] == 'spherical':
                I_disorderedsphere_R0[idx] = res[k]['parameters']['I0']['value']
            else:
                raise ValueError('unhandled population: {}'.format(res[k]))

ds_R0 = pd.read_csv('dataset_R0.csv')
I_pred_R0 = np.array(ds_R0['I0_sphere_pred'],dtype=float).ravel()
I_predvar_R0 = np.array(ds_R0['I0_sphere_pred_var'],dtype=float).ravel()



R1_fpaths = [os.path.join('..','dataset','R1_20190424','R1_20190424_{}_dz_bgsub.yml'.format(idx)) for idx in range(9)]
results_R1 = [yaml.load(open(fpath,'r')) for fpath in R1_fpaths]
I_dilutesphere_R1 = np.zeros(9)
I_precursors_R1 = np.zeros(9)
I_disorderedsphere_R1 = np.zeros(9)

for idx,res in enumerate(results_R1):
    for k in res.keys():
        if not k in ['features','fit_report','noise','sample_metadata']:
            if res[k]['structure'] == 'diffuse' and res[k]['form'] == 'spherical':
                I_dilutesphere_R1[idx] = res[k]['parameters']['I0']['value']
            elif res[k]['structure'] == 'diffuse' and res[k]['form'] == 'guinier_porod':
                I_precursors_R1[idx] = res[k]['parameters']['I0']['value']
            elif res[k]['structure'] == 'disordered' and res[k]['form'] == 'spherical':
                I_disorderedsphere_R1[idx] = res[k]['parameters']['I0']['value']
            else:
                raise ValueError('unhandled population: {}'.format(res[k]))

ds_R1 = pd.read_csv('dataset_R1.csv')
I_pred_R1 = np.array(ds_R1['I0_sphere_pred'],dtype=float).ravel()
I_predvar_R1 = np.array(ds_R1['I0_sphere_pred_var'],dtype=float).ravel()



R2_fpaths = [os.path.join('..','dataset','R2_20190424','R2_20190424_{}_dz_bgsub.yml'.format(idx)) for idx in range(9)]
results_R2 = [yaml.load(open(fpath,'r')) for fpath in R2_fpaths]
I_dilutesphere_R2 = np.zeros(9)
I_precursors_R2 = np.zeros(9)
I_disorderedsphere_R2 = np.zeros(9)

for idx,res in enumerate(results_R2):
    for k in res.keys():
        if not k in ['features','fit_report','noise','sample_metadata']:
            if res[k]['structure'] == 'diffuse' and res[k]['form'] == 'spherical':
                I_dilutesphere_R2[idx] = res[k]['parameters']['I0']['value']
            elif res[k]['structure'] == 'diffuse' and res[k]['form'] == 'guinier_porod':
                I_precursors_R2[idx] = res[k]['parameters']['I0']['value']
            elif res[k]['structure'] == 'disordered' and res[k]['form'] == 'spherical':
                I_disorderedsphere_R2[idx] = res[k]['parameters']['I0']['value']
            else:
                raise ValueError('unhandled population: {}'.format(res[k]))

ds_R2 = pd.read_csv('dataset_R2.csv')
I_pred_R2 = np.array(ds_R2['I0_sphere_pred'],dtype=float).ravel()
I_predvar_R2 = np.array(ds_R2['I0_sphere_pred_var'],dtype=float).ravel()




fig, (ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(20,6))
r_target = np.arange(10.,46.,4.)


ax1.errorbar(r_target,I_pred_R0[:-1],yerr=I_predvar_R0[:-1],label='colloidal particles (pred.)')
ax1.scatter(r_target,I_dilutesphere_R0,c='c',s=30,label='colloidal particles (meas.)')
ax1.scatter(r_target,I_disorderedsphere_R0,c='y',s=30,label='condensed particles')
ax1.scatter(r_target,I_precursors_R0,c='r',s=30,label='unreacted precursors')
ax1.plot(r_target,10.*np.ones(9),'r--',label='design constraint (minimum)')
ax1.set_xlabel('target radius (A)',fontsize=16)
ax1.set_ylabel('integrated intensity (log scale)',fontsize=16)

ax1.set_xticks(r_target)
ax1.set_xlim(left=8.,right=44.)
ax1.set_yscale('log')
ax1.set_ylim(bottom=0.1,top=1000.)
ax1.legend(framealpha=0.5,loc='upper left')
#ax1.spines['left'].set_linewidth(4)
#ax1.spines['right'].set_linewidth(4)
#ax1.spines['top'].set_linewidth(4)
#ax1.spines['bottom'].set_linewidth(4)
ax1.tick_params(labelsize=14)

ax2.errorbar(r_target,I_pred_R1[:-1],yerr=I_predvar_R1[:-1],label='colloidal particles (pred.)')
ax2.scatter(r_target,I_dilutesphere_R1,c='c',s=30,label='colloidal particles (meas.)')
ax2.scatter(r_target,I_disorderedsphere_R1,c='y',s=30,label='condensed particles')
ax2.scatter(r_target,I_precursors_R1,c='r',s=30,label='unreacted precursors')
ax2.plot(r_target,10.*np.ones(9),'r--',label='design constraint (minimum)')
ax2.set_xlabel('target radius (A)',fontsize=16)

ax2.set_xticks(r_target)
ax2.set_xlim(left=8.,right=44.)
ax2.set_yscale('log')
ax2.set_ylim(bottom=0.1,top=1000.)
ax2.tick_params(labelsize=14,labelleft=False)


ax3.errorbar(r_target,I_pred_R2,yerr=I_predvar_R2,label='colloidal particles (pred.)')
ax3.scatter(r_target,I_dilutesphere_R2,c='c',s=30,label='colloidal particles (meas.)')
ax3.scatter(r_target,I_disorderedsphere_R2,c='y',s=30,label='condensed particles')
ax3.scatter(r_target,I_precursors_R2,c='r',s=30,label='unreacted precursors')
ax3.plot(r_target,10.*np.ones(9),'r--',label='design constraint (minimum)')
ax3.set_xlabel('target radius (A)',fontsize=16)

ax3.set_xticks(r_target)
ax3.set_xlim(left=8.,right=44.)
ax3.set_yscale('log')
ax3.set_ylim(bottom=0.1,top=1000.)
ax3.tick_params(labelsize=14,labelleft=False)

plt.savefig('intensity_plots.png')
plt.show()


