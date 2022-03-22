# PlotFunctions Module
#
####################################################################################################
## Import libraries
####################################################################################################
from cProfile import label
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean
from matplotlib.ticker import MultipleLocator

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
    if isinstance(plot_df.index, pd.DatetimeIndex):
        s.index = [plot_df.index[0] - pd.Timedelta(days=7)]
    else:
        s.index = [plot_df.index[0] - 1]
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

#---------------------------------------------------------------------------------------------------
# sr_plot function
#---------------------------------------------------------------------------------------------------
def sr_plot(portfolio_list, names, colors, path=None):
    """Plot of the Sharpe ratio calculated over a rolling 2-year period
    
    Inputs
    portfolio_list: list of portfolio objects corresponding to the backtest of each model
    names: list of strings with the portfolio names that shall appear in the plot legend
    colors: list of strings with matplotlib color names to be used for each portfolio

    Output
    SR evolution figure
    """
    time_period = 104
    df = pd.concat([portfolio_list[i].rets.rets.rename(names[i]) for i in 
                        range(len(portfolio_list))], axis=1)
    mean_df = ((df+1).rolling(time_period).apply(gmean))**52 - 1
    mean_df.dropna(inplace=True)
    std_df = df.rolling(time_period).std()
    std_df.dropna(inplace=True)
    plot_df = mean_df / (std_df * np.sqrt(52))

    fig, ax = plt.subplots(figsize=(6,4))
    for i in range(len(portfolio_list)):
        ax.plot(plot_df[names[i]], color=colors[i])

    ax.legend(names, ncol=3, fontsize=14, loc='upper center', bbox_to_anchor=(0.5, -0.15))
    ax.grid(b="on",linestyle=":",linewidth=0.8)
    ax.tick_params(axis='x', labelrotation = 30)
    plt.ylabel("2-yr SR", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if path is not None:
        fig.savefig(path, bbox_inches='tight')

#---------------------------------------------------------------------------------------------------
# sr_plot function
#---------------------------------------------------------------------------------------------------
def sr_bar(portfolio_list, names, colors, path=None):
    """Plot of the Sharpe ratio calculated over a rolling 2-year period
    
    Inputs
    portfolio_list: list of portfolio objects corresponding to the backtest of each model
    names: list of strings with the portfolio names that shall appear in the plot legend
    colors: list of strings with matplotlib color names to be used for each portfolio

    Output
    SR evolution figure
    """
    n = len(portfolio_list)
    df = pd.concat([portfolio_list[i].rets.rets.rename(names[i]) for i in 
                        range(n)], axis=1)
    mean_df = df.groupby(pd.Grouper(freq='1Y')).mean()
    std_df  = df.groupby(pd.Grouper(freq='1Y')).std()
    plot_df = mean_df / std_df * np.sqrt(52)

    x = np.arange(plot_df.shape[0])
    w = 1/n
    fig, ax = plt.subplots(figsize=(6,4))
    for i in range(n):
        ax.bar(x - 0.5 + i/n, plot_df[names[i]], w, color=colors[i])

    ax.legend(names, ncol=3, fontsize=14, loc='upper center', bbox_to_anchor=(0.5, -0.15))
    ax.grid(b="on",linestyle=":",linewidth=0.8)
    ax.set_xticks(x, plot_df.index.year.to_list())
    
    ax.set_xticks(np.arange(-.6, plot_df.shape[0], 1), minor=True)
    ax.grid(which='minor', color='grey', linestyle='-', linewidth=1)
    ax.grid(which='major', color='w', linestyle='-', linewidth=0)

    plt.ylabel("Annual Sharpe ratio", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if path is not None:
        fig.savefig(path, bbox_inches='tight')
