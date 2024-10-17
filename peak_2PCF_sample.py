import treecorr
import fitsio
import numpy as np
import time
import pprint
import matplotlib
import matplotlib.pyplot as plt
import os


config_DD = {'x_col':1,'y_col':2,                                         #defining configuration file
          'x_units' : 'deg',
          'y_units' : 'deg'}

SNR_dir='SNR_3.5'  #change the SNR cuts
cut=3.5

input_dir  = './euclid_cosmo/fid_a_KS/'             #path of input directories of the KS maps
output_dir = './euclid_cosmo/fid_a_KS/output/'      #path of output directory


###################################################################################################
# Define the folder structure
folders = ["SNR_1","SNR1.5", "SNR_2","SNR2.5", "SNR_3","SNR3.5","SNR_4"]

# Iterate through the folders and create them
for folder in folders:
    
    path = os.path.join(output_dir, folder)
    
    # Create the folder if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"'{folder}' created successfully.")
    else:
        print(f"'{folder}' already exists.")

#################################################################################################
# Define the folder path and directory names
folder_path = output_dir + SNR_dir
directory_name1 = 'correlation'
directory_name2 = 'peak_2PCF'

# Combine folder path and directory names
dir1 = os.path.join(folder_path, directory_name1)
dir2 = os.path.join(folder_path, directory_name2)

# Check if the directories already exist
if not os.path.exists(dir1) or not os.path.exists(dir2):
    # If they don't exist, create them
    os.makedirs(dir1, exist_ok=True)
    os.makedirs(dir2, exist_ok=True)
    
    print(f"Directories '{directory_name1}' and '{directory_name2}' created successfully inside '{folder_path}'.")
else:
    # If they already exist, print a message
    print(f"Directories '{directory_name1}' and '{directory_name2}' already exist inside '{folder_path}'.")

#############################################################################################
#perfrom SNR cuts

for i in range(1,6):
    peak_data = np.loadtxt(f'{input_dir}GalCatalog_LOS_cone1.fits_s1_zmin0.0_zmax3.0.fits_s100{i}_spec0_p0.59_rout10.0_rin0.4_xc0.15_peaksclus_KS.res')
    peak_SNR_cut = peak_data[peak_data[:, 2] >cut]
    np.savetxt(f'{output_dir}{SNR_dir}/snrcut{i}.dat', peak_SNR_cut)


##################################################################################################
#pass random data vector, return curres random catalog and write correaltion objects
def rr_correlation(file_r, i):
    rr = treecorr.NNCorrelation(nbins=15, min_sep=0.01, max_sep=1, sep_units='deg', bin_slop=0)
    cat_r = treecorr.Catalog(file_r, config_DD)
    rr.process(cat_r)
    rr.write(f'{output_dir}{SNR_dir}/correlation/random_correlation_s{i}.dat')
    return cat_r

def dr_correlation(cat_d, cat_r, i):#pass random-data catalog and correaltion
    dr = treecorr.NNCorrelation(nbins=15, min_sep=0.01, max_sep=1, sep_units='deg', bin_slop=0)
    dr.process(cat_d, cat_r)
    dr.write(f'{output_dir}{SNR_dir}/correlation/random_data_correlation_s{i}.dat')

                                   #pass input peak catalog,call treecorr, write the ddcorrelation, return the cat_d
def dd_correlation(file_d, i):                          
    dd = treecorr.NNCorrelation(nbins=15, min_sep=0.01, max_sep=1, sep_units='deg', bin_slop=0)
    cat_d = treecorr.Catalog(file_d, config_DD)
    dd.process(cat_d)
    dd.write(f'{output_dir}{SNR_dir}/correlation/dd_correlation_{i}.dat')
    return cat_d

                                                #pass random data vector, return curres random catalog and write rr correaltion
def rr_correlation(file_r, i):
    rr = treecorr.NNCorrelation(nbins=15, min_sep=0.01, max_sep=1, sep_units='deg', bin_slop=0)
    cat_r = treecorr.Catalog(file_r, config_DD)
    rr.process(cat_r)
    rr.write(f'{output_dir}{SNR_dir}/correlation/random_correlation_{i}.dat')
    return cat_r

def dr_correlation(cat_d, cat_r, i):#pass random-data catalog and correaltion
    dr = treecorr.NNCorrelation(nbins=15, min_sep=0.01, max_sep=1, sep_units='deg', bin_slop=0)
    dr.process(cat_d, cat_r)
    dr.write(f'{output_dir}{SNR_dir}/correlation/random_data_correlation_{i}.dat')


#####################################################################################################

for i in range(1,6):                                                               #iterate through the 5LOS peak catalog
    
    file_d = f'{output_dir}{SNR_dir}/snrcut{i}.dat'
    file_1 = f'{input_dir}GalCatalog_LOS_cone1.fits_s1_zmin0.0_zmax3.0.fits_s100{i}_spec0_p0.59_rout10.0_rin0.4_xc0.15_peaksclus_KS.res'
    #file_N = f'{Noise_dir}GalCatalog_LOS_cone1.fits_s1_zmin0.0_zmax3.0.fits_s1001_spec2_p0.59_rout3.0_rin0.4_xc0.5_peaksclus.res'
    cat_d  = dd_correlation(file_d, i)                                             #create dd-correaltion
    #cat_N  = NN_correlation(file_N, i)
    
    
    n_peaks = len(np.loadtxt(file_1))                                             #create random catalog
    min_x, max_x = 0.01, 10
    random_no_1 = np.random.uniform(min_x, max_x, [10 * n_peaks, 2])
    np.savetxt(f'{output_dir}{SNR_dir}/random_{i}.dat', random_no_1)
    cat_r = rr_correlation(f'{output_dir}{SNR_dir}/random_{i}.dat', i)           #create data-random correaltion object
    dr_correlation(cat_d, cat_r, i)




