import matplotlib.pyplot as plt
import numpy as np
import treecorr
import fitsio
import numpy as np
import time
import pprint
import matplotlib



config_DD = {'x_col':1,'y_col':2,                                         #defining configuration file
          'x_units' : 'deg',
          'y_units' : 'deg'}

SNR_dir='SNR_3.5' #change accordingly
cut=3.5 #change

input_dir  = './euclid_cosmo/fid_a_KS/'  
output_dir = './euclid_cosmo/fid_a_KS/output/'      

#SNR_dir='SNR4'
xi=[]
xi_ls=0
std_err=0
xi_mean=0

for k in range(1,6):
    
    #N_n=len(f'{Noise_dir}GalCatalog_LOS_cone1.fits_s1_zmin0.0_zmax3.0.fits_s1001_spec2_p0.59_rout3.0_rin0.4_xc0.5_peaksclus.res')
    #N_D=len(f'{output_dir}{SNR_dir}/snrcut{i}.dat')
    #frac=N_n/N_D
    
    dd_1 = np.loadtxt(f'{output_dir}{SNR_dir}/correlation/dd_correlation_{k}.dat')
    rr_1 = np.loadtxt(f'{output_dir}{SNR_dir}/correlation/random_correlation_{k}.dat')
    dr_1 = np.loadtxt(f'{output_dir}{SNR_dir}/correlation/random_data_correlation_{k}.dat')

    #nn_1=np.loadtxt(f'{output_dir}{SNR_dir}/correlation/Noise_correlation_s{k}.dat')
    #dn_1=np.loadtxt(f'{output_dir}{SNR_dir}/correlation/data_Noise_correlation_s{i}.dat')
    
    #xi_ls = 1 + ((frac)**2) * (dd_1[:, 3] / nn_1[:, 3]) - frac * (dn_1[:, 3] / nn_1[:, 3])
    xi_ls = 1 + 100 * (dd_1[:, 3] / rr_1[:, 3]) - 10 * (dr_1[:, 3] / rr_1[:, 3])
    np.savetxt(f'{output_dir}{SNR_dir}/peak_2PCF/xi_ls_{k}.dat',xi_ls)
    xi.append(xi_ls)

    plt.plot(rr_1[:, 1], xi_ls, color='grey')

xi_mean = np.mean(np.array(xi), axis=0)
np.savetxt(f'./euclid_cosmo/fid_a_KS/output/xi_mean_{SNR_dir}_KS.txt',xi_mean)
print(xi_mean)
std_err=np.std(np.array(xi), axis=0)
#plt.plot(rr_1[:, 1], np.zeros(15), color='black')
#plt.plot(rr_1[:, 1], xi_mean,label=r'mean $\xi$')
plt.errorbar(rr_1[:, 1], xi_mean,label=r'mean $\xi$', yerr=std_err)
plt.xlabel(r'$\theta$ (degrees)')
plt.ylabel(r'$\xi_{ls}$')
#plt.xscale('log')
#plt.yscale('log')
plt.title('Peak 2PCF: KS maps')
#plt.ylim(-0.001,100)
plt.legend(loc='upper right')
plt.annotate(rf'$\nu > {cut}$', xy=(0.05, 0.05), xycoords='axes fraction', #change
             fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

plt.savefig(output_dir + f'snr_{SNR_dir}.png')
plt.show()
