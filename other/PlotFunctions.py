# PlotFunctions Module
#
# Prepared by: Giorgio Costa (gc2958@columbia.edu)
#
####################################################################################################
## Import libraries
####################################################################################################
import pandas as pd
import matplotlib.pyplot as plt

# Matplotlib parameters
plt.close("all")
plt.rcParams["font.family"] ="serif"
plt.rcParams['axes.xmargin'] = 0

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
        ax.legend(["Nominal", "DRO", "Naive"])
    else:
        ax.legend(["Nominal", "DRO"])

    ax.set_ylabel("Total wealth",fontsize=12)
    ax.grid(b="on",linestyle=":",linewidth=0.8)
    ax.tick_params(axis='x', labelrotation = 30)

    if path is not None:
        fig.savefig(path, bbox_inches='tight')
