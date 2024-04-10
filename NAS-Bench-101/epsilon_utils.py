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
                
                mae = np.abs(preds[0,:]-preds[1,:])
                mean = np.mean(preds, axis=0)
                score = np.nanmean(mae/mean)
                
                scores.append(score)
            # Verify how many architectures got NaN scores
            print(weights, np.sum(np.isnan(scores))/len(scores))
            if np.sum(np.isnan(scores))/len(scores) < 0.1:
                return weights
        window_size -= 1

# def epsilon_main_nlp(n_archs, weights, multi=False):
#     recepie_dic = prepare_recepies()
#     recepies = list(recepie_dic.keys())
#     file_list=os.listdir(logs_dir)
    
#     if multi:
#         log_dir = "train_logs_multi_runs/"
#         accs_min = []
#         accs_max = []
#     else:
#         log_dir = "train_logs_single_run/"
#     accs = []
#     nparams = []
#     score = []
    
#     for i in trange(n_archs):
#         if multi:
#             rec = recepies[i]
#             indices = recepie_dic[rec]
#             # As for the same recepie the metric performance does not change,
#             # we only need to compute it once
#             file = file_list[indices[0]]
#         else:
#             file = file_list[i]
#         log = json.load(open(log_dir + file, 'r'))
#         args = Namespace(**log)

#         # Build the model
#         network = AWDRNNModel(args.model,
#                               ntokens,
#                               args.emsize,
#                               args.nhid,
#                               args.nlayers,
#                               args.dropout,
#                               args.dropouth,
#                               args.dropouti,
#                               args.dropoute,
#                               args.wdrop,
#                               args.tied,
#                               args.recepie,
#                               verbose=False)
#         preds = []
#         for weight in weights:
#             # Initialize
#             prepare_seed(21)
#             def initialize_resnet(m):
#                 if type(m)==MultiLinear:
#                     for par in m.weights_raw:
#                         nn.init.constant_(par, weight)
#                 elif type(m)==CustomRNNCell:
#                     for par in m.parameters():
#                         nn.init.constant_(par, weight)
#                 elif type(m)==nn.modules.linear.Linear:
#                     nn.init.constant_(m.weight, weight)
#                 elif type(m)==nn.modules.container.ParameterList:
#                     for par in m.parameters():
#                         nn.init.constant_(par, weight)
#                 elif type(m)==CustomRNN:
#                     initialize_resnet(m.cell)
#                 elif type(m)==ParameterListWeightDrop:
#                     initialize_resnet(m.module)
#                 elif type(m)==nn.modules.container.ModuleDict:
#                     for sub_m in m:
#                         initialize_resnet(sub_m)
#                 elif type(m)==nn.modules.container.ModuleList:
#                     for sub_m in m:
#                         initialize_resnet(sub_m)
#                 elif type(m)==AWDRNNModel:
#                     initialize_resnet(m.rnns)

#             network.apply(initialize_resnet)
#             network.eval()
#             hidden = network.init_hidden(batch_size, weight)
#             # Take care that embedding is not constant
#             nn.init.uniform_(network.encoder.weight, 0, 1)
#             _, _, raw_output, _ = network(x, hidden=hidden, return_h=True)
#             pred = raw_output[-1][:,:,0].flatten().numpy()
#             pred_min = np.nanmin(pred)
#             pred_max = np.nanmax(pred)
#             pred_norm = (pred - pred_min)/(pred_max - pred_min)
#             preds.append(pred_norm)

#         # Compute the score
#         preds = np.array(preds)
#         preds[np.where(preds==0)] = np.nan
#         mae = np.nanmean(np.abs(preds[0,:]-preds[1,:]))
#         mean = np.nanmean(preds)

#         score.append(mae/mean)
#         nparams.append(args.num_params)
        
#         if multi:
#             # Retrieve 3 seeds test errors
#             acc_run = []
#             for ind in indices:
#                 file = file_list[ind]
#                 log = json.load(open('train_logs_multi_runs/' + file, 'r'))
#                 args = Namespace(**log)
#                 try:
#                     acc_run.append(log['test_losses'][-1])
#                 except: 
#                     acc_run.append(np.nan)
#             accs_mean.append(np.nanmean(acc_run))
#             accs_min.append(np.nanmin(acc_run))
#             accs_max.append(np.nanmax(acc_run))
#         else:
#             try:
#                 accs.append(log['test_losses'][-1])
#             except:
#                 accs.append(np.nan)

#     save_dic = {}
#     if multi:
#         save_dic["accs_min"] = accs_min
#         save_dic["accs_max"] = accs_max
#         save_dic["accs_mean"] = accs
#     else:
#         save_dic["accs"] = accs
#     save_dic["score"] = score
#     save_dic["nparams"] = nparams
    
#     return save_dic

def epsilon_main(data, space_name, searchspace, n_archs, weights, device, args):
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
        if '101' in space_name:
            accs.append(searchspace.get_final_accuracy(uid, args.acc_type, args.trainval)[0])
            accs_min.append(searchspace.get_final_accuracy(uid, args.acc_type, args.trainval)[1])
            accs_max.append(searchspace.get_final_accuracy(uid, args.acc_type, args.trainval)[2]) 
        else:
            accs.append(searchspace.get_final_accuracy(uid, args.acc_type, args.trainval))
        
    save_dic = {}
    if '101' in space_name:
        save_dic["accs_min"] = accs_min
        save_dic["accs_max"] = accs_max
        save_dic["accs_mean"] = accs
    else:
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


def compute_stats(score, accs, reverse=False, raw=False):
    # Take care when the scoring model performance metric should be minimized (perplexity, RMSE)
    if reverse:
        accs = -np.array(accs)
        
    # Take care of NaN entries for accurate stats calculations
    nonan = ~np.isnan(score) & ~np.isnan(accs) & (np.array(score)>0)

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
    plt.xlabel('Test Accuracy', fontsize = 22)
    plt.ylabel('epsilon', fontsize = 22)
    plt.savefig(save_dir + file_name + '.pdf',
                bbox_inches='tight', 
                dpi=300,
                format='pdf')
    plt.show()
##

def bs_ablation_plots(exp_list, name, filename, title):
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
    for exp,ename in exp_list:
        plot_exp(exp,ename)
    plt.grid()
    plt.xlabel('Batch size', fontsize=22)
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

def compute_epsilon(x, network, weights):
    # The initial window spans between 1e-10 and 1e+5
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