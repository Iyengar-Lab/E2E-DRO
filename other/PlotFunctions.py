# PlotFunctions Module
#
# Prepared by: Giorgio Costa (gc2958@columbia.edu)
#
####################################################################################################
## Import libraries
####################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Matplotlib parameters
plt.close("all")
plt.rcParams["font.family"] ="serif"
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['text.usetex'] = True

####################################################################################################
# Ploting functions
####################################################################################################

#---------------------------------------------------------------------------------------------------
# loss_plot function
#---------------------------------------------------------------------------------------------------
def loss_plot(df, path=None):

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(df.loss, color="dodgerblue")
    ax.set_xlabel("Epoch",fontsize=12)
    ax.set_ylabel("Loss",color="dodgerblue",fontsize=12)
    ax2 = ax.twinx()
    ax2.plot(df.val_loss, color="salmon")
    ax2.set_ylabel("Validation loss",color="salmon",fontsize=12)
    ax.grid(b="on",linestyle=":",linewidth=0.8)

    if path is not None:
        fig.savefig(path, bbox_inches='tight')

#---------------------------------------------------------------------------------------------------
# gamma_plot function
#---------------------------------------------------------------------------------------------------
def gamma_plot(df, path=None):

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(df.gamma, color="dodgerblue")
    ax.grid(b="on",linestyle=":",linewidth=0.8)
    ax.set_xlabel("Epoch",fontsize=12)
    if 'delta' in df:
        ax.set_ylabel("gamma",color="dodgerblue",fontsize=12)
        ax2 = ax.twinx()
        ax2.plot(df.delta, color="salmon")
        ax2.set_ylabel("delta",color="salmon",fontsize=12)
    else:
        ax.set_ylabel("gamma",fontsize=12)

    if path is not None:
        fig.savefig(path, bbox_inches='tight')

#---------------------------------------------------------------------------------------------------
# wealth_plot function
#---------------------------------------------------------------------------------------------------
def wealth_plot(nom_df, dro_df, naive_df=None, path=None):

    if naive_df is not None:
        plot_df = pd.concat([nom_df.rets.tri.rename("nom")*100, dro_df.rets.tri.rename("dro")*100, 
                    naive_df.rets.tri.rename("naive")*100], axis=1)
        s = pd.DataFrame({"nom": [100], "dro": [100], "naive": [100]})
    else:
        plot_df = pd.concat([nom_df.rets.tri.rename("nom")*100, dro_df.rets.tri.rename("dro")*100], 
                    axis=1)
        s = pd.DataFrame({"nom": [100], "dro": [100]})

    s.index = [plot_df.index[0]-pd.Timedelta(days=7)]
    plot_df = pd.concat([s, plot_df])

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(plot_df.nom, color="dodgerblue")
    ax.plot(plot_df.dro, color="salmon")

    if naive_df is not None:
        ax.plot(plot_df.naive, color="forestgreen")
        ax.legend(["Nominal E2E", "DR E2E", "Pred.-then-Opt."], fontsize=14)
    else:
        ax.legend(["Nominal E2E", "DR E2E"], fontsize=14)

    ax.grid(b="on",linestyle=":",linewidth=0.8)
    ax.tick_params(axis='x', labelrotation = 30)
    plt.ylabel("Total wealth", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if path is not None:
        fig.savefig(path, bbox_inches='tight')





#---------------------------------------------------------------------------------------------------
# loss_plot_multiple function
#---------------------------------------------------------------------------------------------------
def loss_plot_multiple(nom, dro, path=None):

    K = len(nom)

    fig, axs = plt.subplots(K, 2, figsize=(6.5,18))
    for k in range(K):

        # max_y_train = max([nom[k].loss.max(), dro[k].loss.max()]) + 0.0025
        # min_y_train = min([nom[k].loss.min(), dro[k].loss.min()]) - 0.0025

        # max_y_val = max([nom[k].val_loss.max(), dro[k].val_loss.max()]) + 0.0025
        # min_y_val = min([nom[k].val_loss.min(), dro[k].val_loss.min()]) - 0.0025

        # Nominal plots
        axs[k,0].plot(nom[k].loss, color="dodgerblue")
        axs[k,0].plot(dro[k].loss, color="salmon")
        axs[k,0].grid(b="on",linestyle=":", linewidth=0.8)
        if k < K-1:
            axs[k,0].xaxis.set_ticklabels([])
        if k == 0:
            axs[k,0].set_title(r'Training Loss')
        if k == K-1:
            axs[k,0].set_xlabel(r'Epoch', fontsize=12)

        # DRO loss plots
        l1 = axs[k,1].plot(nom[k].val_loss, color="dodgerblue", label=r'Nominal')
        l2 = axs[k,1].plot(dro[k].val_loss, color="salmon", label=r'DRO')
        axs[k,1].grid(b="on",linestyle=":", linewidth=0.8)
        if k < K-1:
            axs[k,1].xaxis.set_ticklabels([])
        if k == 0:
            axs[k,1].set_title(r'Validation Loss', fontsize=12)
        if k == K-1:
            axs[k,1].set_xlabel(r'Epoch', fontsize=12)
            handles, labels = axs[k,1].get_legend_handles_labels()

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.0), fancybox=True,
                shadow=False, ncol=2, fontsize=12)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), fancybox=True,
                shadow=False, ncol=3, fontsize=12)
    fig.align_ylabels(axs)
    fig.tight_layout()

    if path is not None:
        fig.savefig(path, bbox_inches='tight')

#---------------------------------------------------------------------------------------------------
# gamma_plot_multiple function
#---------------------------------------------------------------------------------------------------
def gamma_plot_multiple(nom, dro, path=None):

    K = len(nom)

    fig, axs = plt.subplots(K, 1, figsize=(5,18))
    for k in range(K):

        # Gamma plots
        axs[k].plot(nom[k].gamma, color="dodgerblue", label=r'Nom. $\displaystyle \gamma$')
        axs[k].set_ylabel(r'$\displaystyle \gamma$', fontsize=16)
        axs[k].plot(dro[k].gamma, color="salmon", label=r'DRO $\displaystyle \gamma$')

        # Delta plot
        ax2 = axs[k].twinx()
        ax2.plot(dro[k].delta, color="salmon", linestyle='--')
        axs[k].plot(np.nan, color="salmon", linestyle='--', label=r'DRO $\displaystyle \delta$')
        ax2.set_ylabel(r'$\displaystyle \delta$', fontsize=16)
        ax2.yaxis.set_label_coords(1.15, 0.5)

        axs[k].grid(b="on",linestyle=":", linewidth=0.8)
        if k < K-1:
            axs[k].xaxis.set_ticklabels([])
        if k == 0:
            axs[k].set_title(r'$\displaystyle \gamma$ and $\displaystyle \delta$ evolution', 
                            fontsize=12)
        if k == K-1:
            axs[k].set_xlabel(r'Epoch', fontsize=12)
            handles, labels = axs[k].get_legend_handles_labels()

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.0), fancybox=True,
                shadow=False, ncol=3, fontsize=12)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), fancybox=True,
                shadow=False, ncol=3, fontsize=12)
    fig.align_ylabels(axs)
    fig.tight_layout()

    if path is not None:
        fig.savefig(path, bbox_inches='tight')

#---------------------------------------------------------------------------------------------------
# wealth_plot function
#---------------------------------------------------------------------------------------------------
def wealth_plot_multiple(nom, dro, path=None):

    K = len(nom)

    fig, axs = plt.subplots(K, 1, figsize=(5,18))
    for k in range(K):
        plot_df = pd.concat([nom[k].rets.tri.rename("nom")*100, dro[k].rets.tri.rename("dro")*100], 
                    axis=1)
        s = pd.DataFrame({"nom": [100], "dro": [100]})

        s.index = [plot_df.index[0]-pd.Timedelta(days=7)]
        plot_df = pd.concat([s, plot_df])

        axs[k].plot(plot_df.nom, color="dodgerblue", label=r'Nominal')
        axs[k].plot(plot_df.dro, color="salmon", label=r'DRO')
        axs[k].set_ylabel("Total wealth",fontsize=12)
        axs[k].grid(b="on",linestyle=":",linewidth=0.8)
        if k < K-1:
            axs[k].xaxis.set_ticklabels([])
        if k == 0:
            axs[k].set_title(r'Wealth evolution', 
                            fontsize=12)
        if k == K-1:
            axs[k].tick_params(axis='x', labelrotation = 30)
            handles, labels = axs[k].get_legend_handles_labels()

    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.0), fancybox=True,
                shadow=False, ncol=3, fontsize=12)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), fancybox=True,
                shadow=False, ncol=3, fontsize=12)
    fig.align_ylabels(axs)
    fig.tight_layout()

    if path is not None:
        fig.savefig(path, bbox_inches='tight')
