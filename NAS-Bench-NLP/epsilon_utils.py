import os
import json

import torch

import numpy as np
import seaborn as sns
from scipy import stats
from argparse import Namespace

import matplotlib.pyplot as plt
import matplotlib.colors as colors

def prepare_seed(rand_seed):
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)
    np.random.seed(rand_seed)
##

def compute_stats(score, accs, reverse=False, raw=False):
    # Take care when the scoring model performance metric should be minimized (perplexity, RMSE)
    if reverse:
        accs = -np.array(accs)

    # Take care of NaN and null entries for accurate stats calculations
    score = np.array(score)
    score[score==0] = np.nan
    nonan = ~np.isnan(score) & ~np.isnan(accs)

    accs_nonan = np.array(accs)[nonan]
    score_nonan = np.array(score)[nonan]

    top10top10_ind = (accs_nonan>np.nanpercentile(accs_nonan, 90)) & (score_nonan>np.nanpercentile(score_nonan, 90))
    top10model_ind = accs_nonan>np.nanpercentile(accs_nonan, 90)
    top10score_ind = score_nonan>np.nanpercentile(score_nonan, 90)

    spearman_all = stats.spearmanr(score_nonan, accs_nonan, nan_policy='omit')[0]
    spearman_best = stats.spearmanr(score_nonan[top10model_ind], accs_nonan[top10model_ind], nan_policy='omit')[0]
    kendall_all = stats.kendalltau(score_nonan, accs_nonan, nan_policy='omit')[0]
    kendall_best = stats.kendalltau(score_nonan[top10model_ind], accs_nonan[top10model_ind], nan_policy='omit')[0]
    top10top10_frac = np.sum(top10top10_ind)/np.sum(top10score_ind)*100

    outputs_raw = [spearman_all,
                   spearman_best,
                   kendall_all,
                   kendall_best,
                   top10top10_frac]

    if len(accs_nonan)>64:
        top64top5_ind = (accs_nonan>np.nanpercentile(accs_nonan, 95)) & (score_nonan>=np.sort(score_nonan)[-64])
        outputs_raw.append(sum(top64top5_ind))
    else:
        outputs_raw.append(np.nan)

    outputs = ["{0:.2f}".format(o) for o in outputs_raw]

    if raw:
        return outputs_raw, len(accs_nonan)
    else:
        return outputs, len(accs_nonan)
##

def plot_results(score, accs, save_dir, save_name, accs_min=None, accs_max=None, nparams=None, top10=False, log_scale=False):
    cmap = sns.cubehelix_palette(start=2.6, rot=0.1, hue=0.7, gamma=0.8, dark=0.1, light=0.85, as_cmap=True)
    clr = np.log10(nparams)
    file_name = save_name
    
    if top10:
        keep = accs<np.nanpercentile(accs, 10)
        score = np.array(score)[keep]
        accs = np.array(accs)[keep]
        clr = clr[keep]
        if accs_min is not None:
            accs_min = np.array(accs_min)[keep]
            accs_max = np.array(accs_max)[keep]
        file_name += "_top10%"
        
    fig = plt.figure(figsize=(6,4.5))
    plt.rc('text', usetex=False)
    ax = fig.add_subplot(111)

    ax.scatter(accs,
               score,
               s=30,
               c=clr,
               cmap=cmap,
               vmin=np.log10(np.min(nparams)),
               vmax=np.log10(np.max(nparams)),
               alpha=0.8
                )

    if accs_min is not None:
        # Add min-max range bars
        plt.errorbar(accs,
                     score,
                     xlolims=accs_min,
                     xuplims=accs_max,
                     zorder=0,
                     fmt="none",
                     linewidth=0.5,
                     ecolor='#8f8f8f'
                     )
    if log_scale:
        ax.set_yscale('log')
        file_name += "_log"
        

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)
    
#     ax.set_ylim(0, 100)

    plt.box(on=None)

    plt.grid(color='#dbdbd9', linewidth=0.5)
    plt.xlabel('Test Perplexity', fontsize = 22)
    plt.ylabel('epsilon', fontsize = 22)
    plt.savefig(save_dir + file_name + '.pdf',
                bbox_inches='tight', 
                dpi=300,
                format='pdf')
    plt.show()
##

def prepare_recepies():
    # Prepare dictionary of indices for multiple runs tests
    file_list=os.listdir("train_logs_multi_runs/")
    r_dic = {}
    for ind in range(len(file_list)):
        file = file_list[ind]
        log = json.load(open('train_logs_multi_runs/' + file, 'r'))
        args = Namespace(**log)
        if args.recepie not in r_dic.keys():
            r_dic[args.recepie]=[ind]
        else:
            r_dic[args.recepie]=r_dic[args.recepie]+[ind]
    return r_dic