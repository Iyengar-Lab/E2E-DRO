# PlotFunctions Module
#
####################################################################################################
## Import libraries
####################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean

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
def wealth_plot(portfolio_list, names, colors, nplots=1, path=None):
    """Plot of the portfolio wealth evolution over time (also known as the 'Total Return Index')

    Inputs
    portfolio_list: list of portfolio objects corresponding to the backtest of each model
    names: list of strings with the portfolio names that shall appear in the plot legend
    colors: list of strings with matplotlib color names to be used for each portfolio
    nplots: Number of subplots into which to distribute the results
    path: Path to which to save the image in pdf format. If 'None', then the image is not saved

    Output
    Wealth evolution figure
    """
    n = len(portfolio_list)
    plot_df = pd.concat([portfolio_list[i].rets.tri.rename(names[i])*100 for i in 
                        range(n)], axis=1)
    s = pd.DataFrame([100*np.ones(n)], columns=names)
    if isinstance(plot_df.index, pd.DatetimeIndex):
        s.index = [plot_df.index[0] - pd.Timedelta(days=7)]
    else:
        s.index = [plot_df.index[0] - 1]
    plot_df = pd.concat([s, plot_df])

    if nplots == 1:
        fig, ax = plt.subplots(figsize=(6,4))
        for i in range(n):
            ax.plot(plot_df[names[i]], color=colors[i])
        ax.legend(names, ncol=n, fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                handlelength=1)
        # ax.legend(names, fontsize=14)
        ax.grid(b="on",linestyle=":",linewidth=0.8)
        ax.tick_params(axis='x', labelrotation = 30)
        plt.ylabel("Total wealth", fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
    else:
        fig, ax = plt.subplots(figsize=(max([6, nplots*4]),4), ncols=nplots)
        for i in range(n):
            j = int(nplots/n * i)
            ax[j].plot(plot_df[names[i]], color=colors[i])
            if j == 0:
                ax[j].set_ylabel("Total wealth", fontsize=14)
            ax[j].tick_params(axis='both', which='major', labelsize=14)

        for j in range(nplots):
            i = int(j * n / nplots)
            k = int((j+1) * n / nplots)
            ax[j].legend(names[i:k], ncol=int(n / nplots), fontsize=12, loc='upper center', 
                        bbox_to_anchor=(0.5, -0.15), handlelength=1)
            # ax[j].legend(names[i:k], fontsize=14)
            ax[j].grid(visible="on",linestyle=":",linewidth=0.8)
            ax[j].tick_params(axis='x', labelrotation = 30)

    if path is not None:
        fig.savefig(path, bbox_inches='tight')
        fig.savefig(path[0:-3]+'ps', bbox_inches='tight', format='ps')

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
        fig.savefig(path[0:-3]+'eps', bbox_inches='tight', format='eps')

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
    
    mean_df = df.expanding(min_periods=1).mean().groupby([df.index.year]).tail(1)
    std_df  = df.expanding(min_periods=1).std().groupby([df.index.year]).tail(1)
    plot_df = mean_df / std_df * np.sqrt(52)

    x = np.arange(plot_df.shape[0])
    w = 1/n
    fig, ax = plt.subplots(figsize=(6,4))
    for i in range(n):
        ax.bar(x - 0.5 + i/n, plot_df[names[i]], w, color=colors[i])

    ax.legend(names, ncol=n, fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
            handlelength=1)
    ax.grid(b="on",linestyle=":",linewidth=0.8)
    ax.set_xticks(x, plot_df.index.year.to_list())
    
    ax.set_xticks(np.arange(-.6, plot_df.shape[0], 1), minor=True)
    ax.grid(which='minor', color='grey', linestyle='-', linewidth=1)
    ax.grid(which='major', color='w', linestyle='-', linewidth=0)

    plt.ylabel("Sharpe ratio", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if path is not None:
        fig.savefig(path, bbox_inches='tight')
        fig.savefig(path[0:-3]+'eps', bbox_inches='tight', format='eps')

#---------------------------------------------------------------------------------------------------
# learn_plot function
#---------------------------------------------------------------------------------------------------
def learn_plot(trained_vals, colors, marker, delta_mark, path=None):
    """Plot of the Sharpe ratio calculated over a rolling 2-year period
    
    Inputs
    trained_vals: pd.Dataframe of learned parameters 
    colors: list of strings with matplotlib color names to be used for each portfolio

    Output
    Plot of learned parameters (gamma as bar, delta as line)
    """

    t, n = trained_vals.shape
    x = np.linspace(0, t-1, num=t)

    fig, ax = plt.subplots(figsize=(6,4))
    ax2 = ax.twinx()
    for i in range(n):
        if i < delta_mark:
            ax.stem(x+i/5, trained_vals.iloc[:,i], colors[i], markerfmt=marker[i],
            bottom=trained_vals.iloc[0,i])
        else:
            ax2.stem(x+i/5, trained_vals.iloc[:,i], colors[i], markerfmt=marker[i],
            bottom=trained_vals.iloc[0,i])

    ax.legend(trained_vals.columns, ncol=n, fontsize=12, loc='upper center', 
            bbox_to_anchor=(0.5, -0.15), handlelength=1)
    ax.grid(b="on",linestyle=":",linewidth=0.8)

    ax.set_xlabel(r'Training period', fontsize=14)
    ax.set_ylabel(r'$\gamma$', fontsize=14)
    if i < delta_mark:
        ax2.set_ylabel(r'$\delta$', fontsize=14)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if path is not None:
        fig.savefig(path, bbox_inches='tight')
        fig.savefig(path[0:-3]+'eps', bbox_inches='tight', format='eps')
        
####################################################################################################
# Other results
####################################################################################################
#---------------------------------------------------------------------------------------------------
# fin_table
#---------------------------------------------------------------------------------------------------
def fin_table(portfolios:list, names:list) -> pd.DataFrame:
    """Compute portfolio performance statistics and summarize them as a table
    
    Inputs
    List of backtest-type objects
    
    Outputs
    Table of results    
    """
    
    rets =[]
    vols = []
    SRs = []
    invHidxs = []
    
    for portfolio in portfolios:
        ret = (portfolio.rets.tri.iloc[-1] ** 
                (1/portfolio.rets.tri.shape[0]))**52 - 1
        vol = portfolio.vol * np.sqrt(52)
        SR = ret / vol
        invHidx = round(1/(pd.DataFrame(portfolio.weights) ** 2).sum(axis=1).mean(), ndigits=2)
        rets.append(round(ret*100, ndigits=1))
        vols.append(round(vol*100, ndigits=1))
        SRs.append(round(SR, ndigits=2))
        invHidxs.append(invHidx)

    table  = pd.DataFrame(np.array([rets, vols, SRs, invHidxs]), 
                                   columns=names)
    table.set_axis(['Return (%)', 
                    'Volatility (%)', 
                    'Sharpe ratio',
                    'Avg. inv. HHI'], 
                   axis=0, inplace=True) 
    
    return table
