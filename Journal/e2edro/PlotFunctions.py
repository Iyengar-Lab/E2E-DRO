# PlotFunctions Module
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
# wealth_plot function
#---------------------------------------------------------------------------------------------------
def wealth_plot(portfolio_list, names, colors, path=None):
    """Plot of the portfolio wealth evolution over time (also known as the 'Total Return Index')

    Inputs
    portfolio_list: list of portfolio objects corresponding to the backtest of each model
    names: list of strings with the portfolio names that shall appear in the plot legend
    colors: list of strings with matplotlib color names to be used for each portfolio

    Output
    Wealth evolution figure
    """

    plot_df = pd.concat([portfolio_list[i].rets.tri.rename(names[i])*100 for i in 
                        range(len(portfolio_list))], axis=1)
    s = pd.DataFrame([100*np.ones(len(portfolio_list))], columns=names)
    s.index = [plot_df.index[0]-pd.Timedelta(days=7)]
    plot_df = pd.concat([s, plot_df])

    fig, ax = plt.subplots(figsize=(6,4))
    for i in range(len(portfolio_list)):
        ax.plot(plot_df[names[i]], color=colors[i])

    ax.legend(names, fontsize=14)
    ax.grid(b="on",linestyle=":",linewidth=0.8)
    ax.tick_params(axis='x', labelrotation = 30)
    plt.ylabel("Total wealth", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if path is not None:
        fig.savefig(path, bbox_inches='tight')