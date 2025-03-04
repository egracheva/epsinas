import os
import json

import torch
import torch.nn as nn

import numpy as np
import seaborn as sns
from scipy import stats
from tqdm import trange
from argparse import Namespace

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator

import models

def prepare_seed(rand_seed):
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)
    np.random.seed(rand_seed)
##

# Optimising the weights
def search_weights(x, searchspace, device):
    LOW = -7
    HIGH = 0
    window_size = HIGH-LOW
    while window_size != 0:
        np.random.seed(21)
        for w in range(LOW, HIGH+1-window_size):
            weights = [10**w, 10**(w+window_size)]
            scores = []
            # Keep the same random set of architectures across tested weights
            np.random.seed(21)
            for i in np.random.choice(len(searchspace), 100):
                preds = []
                uid = searchspace[i]
                network = searchspace.get_network(uid)
                network = network.to(device)
                for weight in weights:
                    torch.cuda.empty_cache()
                    # Initialize
                    prepare_seed(21)
                    def initialize_resnet(m):
                        if type(m) == torch.nn.Sequential:
                            for sub_m in m:
                                initialize_resnet(sub_m)
                        else:
                            fill_bias = False
                            if hasattr(m, 'bias'):
                                if m.bias is not None:
                                    fill_bias = True
                            if fill_bias:
                                torch.nn.init.constant_(m.bias, 0)
                            fill_weight = False
                            if hasattr(m, 'weight'):
                                fill_weight = True
                            if hasattr(m, 'affine'):
                                if not m.affine:
                                    fill_weight = False
                            if fill_weight:
                                torch.nn.init.constant_(m.weight, weight)
                    
                    network.apply(initialize_resnet)
                    y_pred, _ = network(x)
                    pred = y_pred.cpu().detach().numpy().flatten()
                    pred[np.where(np.abs(pred)==np.inf)] = np.nan
                    pred_min = np.nanmin(pred)
                    pred_max = np.nanmax(pred)
                    pred_norm = (pred - pred_min)/(pred_max - pred_min)
                    preds.append(pred_norm)

                preds = np.array(preds)
                preds[np.where(preds==0)] = np.nan
                mae = np.nanmean(np.abs(preds[0,:]-preds[1,:]))
                mean = np.nanmean(preds)
                score = mae/mean
                
                scores.append(score)
            # Verify how many architectures got NaN scores
            print(weights, np.sum(np.isnan(scores))/len(scores))
            if np.sum(np.isnan(scores))/len(scores) < 0.1:
                return weights
        window_size -= 1
##

def epsinas_main(data, space_name, searchspace, n_archs, weights, device, args):
    if '101' in space_name:
        accs_min = []
        accs_max = []
    accs = []
    nparams = []
    score = []
    for i in trange(n_archs):
        uid = searchspace[i]
        network = searchspace.get_network(uid)
        network = network.to(device)
        preds = []
        for weight in weights:
            torch.cuda.empty_cache()
            prepare_seed(21)
            # Initialize
            def initialize_resnet(m):
                if type(m) == torch.nn.Sequential:
                    for sub_m in m:
                        initialize_resnet(sub_m)
                else:
                    fill_bias = False
                    if hasattr(m, 'bias'):
                        if m.bias is not None:
                            fill_bias = True
                    if fill_bias:
                        torch.nn.init.constant_(m.bias, 0)
                    fill_weight = False
                    if hasattr(m, 'weight'):
                        fill_weight = True
                    if hasattr(m, 'affine'):
                        if not m.affine:
                            fill_weight = False
                    if fill_weight:
                        torch.nn.init.constant_(m.weight, weight)

            network.apply(initialize_resnet)
            y_pred, _ = network(data)
            pred = y_pred.cpu().detach().numpy().flatten()
            pred_min = np.nanmin(pred)
            pred_max = np.nanmax(pred)
            pred_norm = (pred - pred_min)/(pred_max - pred_min)
            preds.append(pred_norm)

        # Compute the score
        preds = np.array(preds)
        preds[np.where(preds==0)] = np.nan
        mae = np.nanmean(np.abs(preds[0,:]-preds[1,:]))
        mean = np.nanmean(preds)
        
        score.append(mae/mean)
        nparams.append(sum(p.numel() for p in network.parameters()))
        accs.append(searchspace.get_final_accuracy(uid, args.acc_type, args.trainval))
        
    save_dic = {}
    save_dic["accs"] = accs
    save_dic["score"] = score
    save_dic["nparams"] = nparams
    
    return save_dic
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
##

def steiger_z_test(xy, xz, yz, n, conf_level=0.95):
    """
    Function for calculating the statistical significant differences between
    two dependent correlation coefficients.
    Adopted from the R package http://personality-project.org/r/html/paired.r.html
    and is described in detail in the book 'Statistical Methods for Psychology'
    Credit goes to the authors of above mentioned packages!
    Author: Philipp Singer (www.philippsinger.info)
    #copied from on 4/24/2015 from https://github.com/psinger/CorrelationStats/blob/master/corrstats.py

    Calculates the statistic significance between two dependent correlation coefficients
    @param xy: correlation coefficient between x and y
    @param xz: correlation coefficient between x and z
    @param yz: correlation coefficient between y and z
    @param n: number of elements in x, y and z
    @param twotailed: whether to calculate a one or two tailed test, only works for 'steiger' method
    @param conf_level: confidence level, only works for 'zou' method
    @param method: defines the method uses, 'steiger' or 'zou'
    @return: t and p-val
    """
    
    d = xy - xz
    determin = 1 - xy * xy - xz * xz - yz * yz + 2 * xy * xz * yz
    av = (xy + xz)/2
    cube = (1 - yz) * (1 - yz) * (1 - yz)

    t2 = d * np.sqrt((n - 1) * (1 + yz)/(((2 * (n - 1)/(n - 3)) * determin + av * av * cube)))
    p = 1 - stats.t.cdf(abs(t2), n - 2)
    
    # For two-tailed p-value
    p *= 2

    return t2, p
##

def compute_epsinas(x, network, weights):
    preds = []
    for weight in weights:
        torch.cuda.empty_cache()
        
        def initialize_weights(m):
            
            fill_bias = False
            if hasattr(m, 'bias'):
                if m.bias is not None:
                    fill_bias = True

            if fill_bias:
                torch.nn.init.constant_(m.bias, 0)

            fill_weight = False
            
            if hasattr(m, 'weight'):
                fill_weight = True

            if hasattr(m, 'affine'):
                if not m.affine:
                    fill_weight = False

            if fill_weight:
                torch.nn.init.constant_(m.weight, weight)

        prepare_seed(21)
        network.apply(initialize_weights)
        y_pred, _ = network(x)
        pred = y_pred.cpu().detach().numpy().flatten()
        pred_min = np.nanmin(pred)
        pred_max = np.nanmax(pred)
        pred_norm = (pred - pred_min)/(pred_max - pred_min)
        preds.append(pred_norm)

    # Compute the score
    preds = np.array(preds)
    preds[np.where(preds==0.)] = np.nan
    mae = np.abs(preds[0,:]-preds[1,:])
    score = np.nanmean(mae)/np.nanmean(preds)
    
    return score
##


def compute_stats(score, accs, reverse=False, raw=False):
    # Take care when the scoring model performance metric should be minimized (perplexity, RMSE)
    if reverse:
        accs = -np.array(accs)

    # Take care of NaN entries for accurate stats calculations
    nonan = ~np.isnan(score) & ~np.isnan(accs)# & (np.array(score)>0)

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

    plt.box(on=None)

    plt.grid(color='#dbdbd9', linewidth=0.5)
    plt.xlabel('Test Accuracy', fontsize = 22)
    plt.ylabel('epsinas', fontsize = 22)
    plt.savefig(save_dir + file_name + '.pdf',
                bbox_inches='tight', 
                dpi=300,
                format='pdf')
    plt.show()
##

def bs_ablation_plots(exp_list, name, filename, dataset):
    fig = plt.figure(figsize=(7.2,4.45))
    ax = fig.add_subplot(111)
    
    def plot_exp(exp, label):
        exp = np.array(exp)
        q_75 = np.nanquantile(exp, .75, axis=1)
        q_25 = np.nanquantile(exp, .25, axis=1)
        mean = np.nanmedian(exp, axis=1)
        ax.plot(range(len(mean)), mean, label=label)
        ax.fill_between(range(len(mean)), np.nanmin(exp, axis=1), np.nanmax(exp, axis=1), alpha=0.1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xticklabels(['','8','16','32','64','128','256','512','1024'])
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(16)
            
    for exp, ename in exp_list:
        plot_exp(exp, ename)
    plt.grid()
    plt.xlabel('Batch size', fontsize=22)
    
    if dataset=='cifar10':
        title = 'CIFAR-10'
    elif dataset=='cifar100':
        title = 'CIFAR-100'
    else:
        title = dataset
        
    if name=='rho':
        plt.ylabel(r'Spearman $\rho$', fontsize=22)
    elif name=='tau':
        plt.ylabel(r'Kendall $\tau$', fontsize=18)
        plt.title(title, fontsize=24)
    plt.legend(loc=4, fontsize=20)
    plt.savefig(filename,
                bbox_inches='tight', 
                dpi=300,
                format='pdf')
    plt.show()
##