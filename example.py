import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from scipy.linalg import sqrtm
import yfinance as yf

from cvxpylayers.torch import CvxpyLayer



torch.set_default_dtype(torch.double)

# get returns
df = yf.download("VTI IWM MUB GLD SLV LQD SHY")
df = df["Adj Close"].pct_change()
df = df[~df.isna().any(axis=1)]
(1 + df).cumprod(axis=0).plot()
plt.semilogy()
n = df.shape[1]

# get vix
aux = yf.download("^VIX")
aux = aux.loc[df.index]["Adj Close"].shift(-1)
aux = aux.iloc[:-1]
df = df.iloc[:-1]


vix_codes = pd.qcut(aux, 5).cat.codes
P = np.zeros((5, 5))
for i in range(vix_codes.size - 1):
    P[vix_codes.iloc[i], vix_codes.iloc[i+1]] += 1
P /= P.sum(axis=1)[:, None]
plt.imshow(P, vmin=0, vmax=1)
plt.colorbar()

models = {}
for i in range(5):
    log1pr = np.log1p(df[vix_codes == i])
    mu, Sigma = np.array(log1pr.mean()), np.array(log1pr.cov())
    mean = np.exp(mu + .5 * np.diag(Sigma))
    cov = np.diag(mean) @ (np.exp(Sigma) - np.outer(np.ones(n), np.ones(n))) @ np.diag(mean)
    models[i] = (mu, Sigma, mean, cov)

np.set_printoptions(precision=4, suppress=True)
rets = np.array([df[vix_codes == i].mean() * 250 for i in range(5)])
stds = np.array([df[vix_codes == i].std() * np.sqrt(250) for i in range(5)])

vix_state = 0
R = []
for _ in range(252*5):
    mean, cov, _, _ = models[vix_state]
    R.append(np.exp(np.random.multivariate_normal(mean, cov)))
    vix_state = np.random.choice(np.arange(5), p=P[vix_state])
R = np.array(R)

n = df.shape[1]

wtilde = cp.Variable(n)
u = cp.Variable(n)
f = cp.Variable(n)

w = cp.Parameter(n)
alpha = cp.Parameter(n)
S = cp.Parameter((n, n))
s = cp.Parameter(1, nonneg=True)
kappa = cp.Parameter(1, nonneg=True)
gamma = cp.Parameter(1, nonneg=True)

expected_return = alpha @ wtilde
expected_risk = cp.sum_squares(f)
holding_cost = s * cp.sum(cp.neg(wtilde))
transaction_cost = kappa * cp.sum(cp.abs(u))

constraints = [
    cp.sum(wtilde) == 1,
    cp.norm(wtilde, 1) <= 1.5,
    cp.norm(wtilde, "inf") <= .5,
    f == S @ wtilde,
    u == wtilde - w
]

prob = cp.Problem(
    cp.Maximize(expected_return - gamma * expected_risk - holding_cost - transaction_cost),
    constraints
)

trading_policy = CvxpyLayer(prob, [w, alpha, S, s, kappa, gamma], [wtilde, u, f])


for i in range(5):
    _, _, mean, cov = models[i]
    w_tmp, _, _ = trading_policy(torch.zeros(n, dtype=torch.float32) / n, torch.from_numpy(mean).float(), torch.from_numpy(sqrtm(cov)).float(), torch.ones(1, dtype=torch.float32) * .0002, torch.zeros(1, dtype=torch.float32), torch.ones(1, dtype=torch.float32) * 5,
                                solver_args={"solve_method": "ECOS", "n_jobs_forward": 1})
    print(w_tmp.numpy())


def my_fun(xx):

    return xx + n

def simulation(T, batch_size, params, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    alpha_tch, S_tch, s_tch, kappa_tch, gamma_tch = params
    w_tch, _, _ = trading_policy(torch.zeros(batch_size, n) / n, alpha_tch[0].float(), S_tch[0].float(),
                                 s_tch[0].float(), 0.*kappa_tch[0].float(), gamma_tch[0].float(),
                                 solver_args={"solve_method": "ECOS", "n_jobs_forward": 1})
    v = torch.ones(batch_size, 1)
    returns = torch.zeros(batch_size, 0)
    ws = []
    us = []
    vix_states = [0]*batch_size

    for t in range(T-1):
        ws.append(w_tch.detach().numpy())
        wtilde, u, _ = trading_policy(w_tch, alpha_tch[vix_states,:].float(), S_tch[vix_states,:,:].float(),
                                      s_tch[vix_states, :].float(), kappa_tch[vix_states, :].float(), gamma_tch[vix_states, :].float(),
                                     solver_args={"solve_method": "ECOS", "n_jobs_forward": 1})
        us.append(u.detach().numpy())
        wtilde /= wtilde.sum(1).unsqueeze(1)
        u = wtilde - w_tch
        r = []
        for j in range(batch_size):
            mu, Sigma, _, _ = models[vix_states[j]]
            r += [np.exp(np.random.multivariate_normal(mu, Sigma)) - 1]
        r = torch.from_numpy(np.array(r))
        tc = .001 * u.abs().sum(1)
        sc = .0002 * (wtilde * (wtilde <= 0)).sum(1)
        total_return = 1 + (wtilde * r).sum(1) - tc - sc
        w_tch = wtilde * (1 + r)
        w_tch /= w_tch.sum(1).unsqueeze(1)
        v = torch.cat([v, (v[:, t] * total_return).unsqueeze(-1)], axis=1)
        returns = torch.cat([returns, total_return.unsqueeze(-1)], axis=1)
                
        for j in range(batch_size):
            vix_states[j] = np.random.choice(np.arange(5), p=P[vix_states[j]])

    realized_return = 100 * 252 * (returns - 1).mean()
    realized_volatility = 100 * np.sqrt(252) * (returns - 1).pow(2).mean(1).sqrt().mean()
    average_drawdown = 100 * (torch.cummax(v, 1).values / v - 1).mean()

    objective = -realized_return / realized_volatility + average_drawdown
    
    return objective, realized_return, realized_volatility, average_drawdown, v, np.array(ws), np.array(us)



T = 250*3
validation_seed = 538
batch_size = 8

s_tch = torch.log(torch.ones(5, 1) * .0002)
kappa_tch = torch.log(torch.ones(5, 1) * .001)
gamma_tch = torch.log(torch.ones(5, 1) * 5)

s_tch.requires_grad_(True)
kappa_tch.requires_grad_(True)
gamma_tch.requires_grad_(True)

alpha_tch = torch.from_numpy(np.array([np.array(df.mean())] * 5))
S_tch = torch.from_numpy(np.array([sqrtm(np.array(df.cov()))] * 5))

with torch.no_grad():
    params = [alpha_tch, S_tch, torch.exp(s_tch), torch.exp(kappa_tch), torch.exp(gamma_tch)]
    objective, ret, risk, drawdown, v0_common, w0_common, u0_common = simulation(T, batch_size, params, seed=validation_seed)
    print (objective.item(), ret.item(), risk.item(), drawdown.item())