import treecorr
import fitsio
import numpy as np
import time
import pprint
import matplotlib
import matplotlib.pyplot as plt
import os
import h5py


config_DD = {'x_col':1,'y_col':2,                                         #defining configuration file
          'x_units' : 'deg',
          'y_units' : 'deg'}

input_dir  = './euclid_cosmo/peaks_clustering_KS_kappa_maps_from_MEAN_shear_maps/KS_maps/outputs_v2/peaks/DVs/' 
output_dir = './euclid_cosmo/peaks_clustering_KS_kappa_maps_from_MEAN_shear_maps/output/'      #output fil   #noise file
cut=1
SNR_dir="SNR_1"

# Define the folder structure
folders = ["SNR_1", "SNR_2", "SNR_3","SNR_4"]

# Iterate through the folders and create them
for folder in folders:
    
    path = os.path.join(output_dir, folder)
    
    # Create the folder if it doesn't exist
    if not os.path.exists(path):a
        os.makedirs(path)
        print(f"'{folder}' created successfully.")
    else:
        print(f"'{folder}' already exists.")


for i in range(1,6):
    peak_data = np.loadtxt(f'{input_dir}GalCatalog_LOS_cone1.fits_s1_zmin0.0_zmax3.0.fits_s1004_spec0_p0.59_KS_kappamap_kE_gaussian_1.71_px.res_peak_catalogue.res')
    peak_SNR_cut= peak_data[peak_data[:, 2] > cut]
    np.savetxt(f'{output_dir}{SNR_dir}/snr_test{i}.dat', peak_SNR_cut)


dd = treecorr.NNCorrelation(nbins=15, min_sep=0.01, max_sep=1, sep_units='deg', bin_slop=0)
rr = treecorr.NNCorrelation(nbins=15, min_sep=0.01, max_sep=1, sep_units='deg', bin_slop=0)
dr = treecorr.NNCorrelation(nbins=15, min_sep=0.01, max_sep=1, sep_units='deg', bin_slop=0)

for k in range(1,6):
    cat_d  = treecorr.Catalog(f'{output_dir}{SNR_dir}/snr_test{k}.dat', config_DD)
    dd.process(cat_d)
    dd.write(f'{output_dir}{SNR_dir}/dd_{k}.dat')

    min_x, max_x = 0.1, 10
    n_peaks=int(len(np.loadtxt(f'{output_dir}{SNR_dir}/snr_test{k}.dat')))
    print(n_peaks)
    random_1 = np.random.uniform(min_x, max_x, [10 * n_peaks, 2])
    np.savetxt(f'{output_dir}{SNR_dir}/random_{k}.dat',random_1)
    #print(random_1)

    cat_r = treecorr.Catalog(f'{output_dir}{SNR_dir}/random_{k}.dat', config_DD)
    rr.process(cat_r)
    rr.write(f'{output_dir}{SNR_dir}/rr_{k}.dat')
    
    dr.process(cat_d, cat_r)
    dr.write(f'{output_dir}{SNR_dir}/dr_{k}.dat')


xi=[]
for k in range(1,6):
    dd_val=np.loadtxt(f'{output_dir}{SNR_dir}/dd_{k}.dat')
    rr_val=np.loadtxt(f'{output_dir}{SNR_dir}/rr_{k}.dat')
    dr_val=np.loadtxt(f'{output_dir}{SNR_dir}/dr_{k}.dat')
    
    xi_ls = 1 + 100 * (dd_val[:, 3] / rr_val[:, 3]) - 10 * (dr_val[:, 3] / rr_val[:, 3])
    xi.append(np.array(xi_ls))
    plt.plot(rr_val[:, 1],xi_ls)
    plt.yscale('log')
mean_xi=np.mean(xi,axis=0)
std_err=np.std(xi, axis=0)
plt.plot(rr_val[:, 1],mean_xi,'--', label='mean')
plt.xlim(0.1,1)
plt.errorbar(rr_val[:, 1], mean_xi,label=r'mean $\xi$', yerr=std_err)
plt.xlabel(r'$\theta$ (degrees)')
plt.ylabel(r'$\xi_{ls}$')
plt.yscale('log')
plt.title('Peak 2PCF: KS maps')
#plt.legend(loc='upper right')
plt.annotate(rf'$\nu > {cut}$', xy=(0.05, 0.05), xycoords='axes fraction', #change
             fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.legend(loc='upper right')
plt.savefig(f'{output_dir}/{SNR_dir}.png')
plt.show()            