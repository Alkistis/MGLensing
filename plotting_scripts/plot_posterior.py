import numpy as np
import getdist.plots
import matplotlib.pyplot as plt
import yaml
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
with open("plotting_scripts/params_names.yaml", "r") as file:
    params_dic = yaml.safe_load(file)
with open("params_data.yaml", "r") as file:
    fiducials = yaml.safe_load(file)



def read_last_header_line(file_path):
    last_header = None
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith('#'):
                last_header = line.strip('#').strip()  
            else:
                break  

    if last_header:
        return last_header.split() 
    else:
        return []

file_paths = [ 'chains/chain_lsst_y1_3x2pt_bacco_heft_data_b1_hmcode_model_kmax0p1.txt'   
     #'chains/lsst/chain_lsst_y1_3x2pt_bacco_heft_same_data_model.txt',
              #'chains/lsst/chain_lsst_y1_3x2pt_bacco_heft_data_b1_model_lincut.txt',
              #'chains/lsst/chain_lsst_y1_3x2pt_bacco_heft_data_b1_model_kmax0p3.txt',
              #'chains/chain_lsst_y1_3x2pt_bacco_heft_data_b1_model_kmax0p1.txt'
 ]  

file_name = 'hmcode_check'
legend_labels = [
    #'before',
    #'after'
#'model: heft, GG/GL cuts: kmax=0.7 h/Mpc',
'model: b1+hmcode, GG/GL cuts: kmax=0.1 h/Mpc',
#'model: b1, GG/GL cuts: kmax=0.3 h/Mpc'
]
annotation_text = 'LSST Y1 data: bacco+heft\n 3x2pt-analysis\n LL-cuts lmax=2800-3000'
# annotation square
num = 1

n_samples = len(file_paths)
chains_info = {}
for i in range(n_samples):
    file_path = file_paths[i]  
    chain = np.genfromtxt(file_path)
    chain_pars = read_last_header_line(file_path)
    chain_pars = chain_pars[:-2]
    chains_info[i] = {}
    chains_info[i]['chain'] = chain
    chains_info[i]['pars'] = chain_pars


samples = []    
for i in range(n_samples):
    samples.append(
        getdist.MCSamples(samples = chains_info[i]['chain'][:,:len(chains_info[i]['pars'])],
                                    names = [i for i in chains_info[i]['pars']],
                                    weights=np.exp(chains_info[i]['chain'][:, -2]),
                                    labels = [params_dic[p]
                                                for p in chains_info[i]['pars']],
                                    settings={'smooth_scale_2D':0.35, 'smooth_scale_1D':0.35},
                                    ) 
    )


ModelPars = chains_info[0]['pars']
# add parameters that are not present in the first chain:
#ModelPars.append('beta_IA')
ModelPars = ModelPars[:8]


colors = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:olive', 'tab:cyan']
from matplotlib import rc
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Times']})
g = getdist.plots.getSubplotPlotter(subplot_size=1.3)

plt.rcParams.update({'font.size':16})
g.settings.legend_fontsize=26#36
g.settings.axes_fontsize=18#25
g.settings.axes_labelsize=20#32
g.settings.linewidth=3   
g.settings.figure_legend_frame = False


g.triangle_plot(samples,
    ModelPars,
legend_labels = legend_labels,
legend_loc = 'upper right',
contour_args = [{'filled':True, 'color': colors[0]}, {'filled':True, 'color': colors[1], 'ls': '-'}, {'filled':True, 'color': colors[2], 'ls': '-'},
                {'filled':True, 'color': colors[3], 'ls': '-'},  {'filled':True, 'color': colors[4], 'ls': '-'},  {'filled':True, 'color': colors[5], 'ls': '-'}], 
line_args=[ {'color': colors[0], 'ls': '-'}, {'color': colors[1], 'ls': '-'}, {'color': colors[2], 'ls': '-'}, 
            {'color': colors[3], 'ls': '-'}, {'color': colors[4], 'ls': '-'}, {'color': colors[5], 'ls': '-'}])



if fiducials != None:
    for i in range(len(ModelPars)):
        for j in range(i+1):
            ax = g.subplots[i,j]
            if i != j and ModelPars[i] in fiducials and fiducials[ModelPars[i]] != None:
                ax.axhline(fiducials[ModelPars[i]],lw=2.,color='tab:gray')
            if ModelPars[j] in fiducials and fiducials[ModelPars[j]] != None:
                ax.axvline(fiducials[ModelPars[j]],lw=2.,color='tab:gray')

ax = g.subplots[num, num]
ax.annotate(annotation_text, (2.5, 0.05), xycoords='axes fraction', clip_on=False, fontsize=20) 
                

plt.savefig('figs/posteriors/'+file_name+'.png')  

