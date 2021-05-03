import os
os.chdir(os.path.dirname(__file__))
from numpy import maximum, minimum, log, exp, power, sqrt, pi, linspace, meshgrid, frompyfunc, inf, arange
from scipy.stats import norm
from numpy.random import normal
import matplotlib.pyplot as plt
#plt.rcParams['mathtext.fontset'] = 'cm'
#plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = '8'
plt.style.use('ggplot')

gamma, sigma, mu, r, rho, a0, k0, d0, T = 3, 0.2, 0.06, 0.025, 0.02, 100, 95, 90, 10

ce = ((1. - gamma) * a0 ** (1. - gamma) / (1. - gamma) * exp((1. - gamma) * T * (r + (mu - r) ** 2 / (2. * gamma * sigma ** 2)))) ** (1. / (1. - gamma))

n_steps = 10000
dt = 1. * T / n_steps 
t = dt * arange(n_steps + 1)
d = d0 * exp(rho * t)
k = k0 * exp(rho * t)

def paths(idx=0):
    epochs = 50000
    for l in range(epochs):
        path, path1 = [], []
        a = a0
        i1 = -1 
        b_k = False
        for i in range(n_steps + 1):
            w = 1 if b_k else 1
            factor = exp((r + w * (mu - r) - (w * sigma) ** 2 / 2) * dt + w * sigma * sqrt(dt) * normal(0, 1, 1)[0])
            
            if a * factor - k[i] < 1e-4:
                if b_k:
                    if a * factor - d[i] >= 1e-4:
                        a *= factor
                        path1.append(a)
                    else:
                        path1.append(d[i])
                        break
                else:
                    a = (1.05 if idx == 1 else 1) * k[i]
                    path1 = [a]
                    i1 = i
                    b_k = True            
            else:
                a *= factor
                if b_k:
                    path1.append(a)
                else:
                    path.append(a)
        
        width = len(path) + len(path1) > 0.7 * n_steps
        height = max(path + path1) < 1.6 * max(d) 
        path_upper = len(path) < 0.7 * n_steps
        path_lower = len(path) > 0.2 * n_steps
        path1_lower = len(path1) > 0.2 * n_steps
        if idx == 0:
            if width & height & (len(path1) == 0):
                break
        elif idx == 1:
            if width & height & path_upper & path_lower & path1_lower:
                break
        else:
            if width & height & path_upper & path_lower & path1_lower & (len(path) + len(path1) > 0.99 * n_steps):
                break

    return path, path1, i1

fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
for ii, ax in enumerate(axes):
    path, path1, i1 = paths(ii)
    ax.plot(t, d, linestyle=(0, (5, 1)), color='orangered')
    ax.plot(t, k, linestyle=(0, (5, 1)), color='yellow')
    ax.plot(dt * arange(len(path)), path, label='before', color='green')
    if path1:
        ax.plot(dt * arange(i1, i1 + len(path1)), path1, label='after', color='blue')
        ax.vlines(dt * i1, -1000, 1000, linestyle=(0, (5, 1)))#linestyle=(0, (3, 1, 1, 1)))
    if ii == 2:
        ax.set_xlabel('Years')
    ax.set_ylabel('Asset Value')
    if ii:
        ax.legend(loc='lower right', shadow=True)
    ax.set_ylim([0.65 * min(d), 1.1 * max(path + path1)])

fig.tight_layout()
fig.savefig('pm.eps', bbox_inches='tight')

import sys; sys.exit()

alpha = 0.95; l0 = alpha * a0 
beta = 0.
epsilon = 1 - power(1 - 0.005, T)
eps = 0.1

def mu_tilde(w):
    return r + w * (mu - r) - rho - power(sigma * w, 2) / 2 

def f(t0, t1, p0, p1, w):
    return -log(p1 / p0) / (sigma * w * power(t1 - t0, 3. / 2)) / sqrt(2 * pi) * exp(-1 / 2 * ((log(p1 / p0) - mu_tilde(w) * (t1 - t0)) / (sigma * w * sqrt(t1 - t0))) ** 2)
    
def gg(y, t0, t1, p0, p1, w):
    return 1. / (sigma * w * sqrt(t1 - t0)) / sqrt(2 * pi) * exp(-1. / 2 * (power(y, 2) - 2 * mu_tilde(w) * (t1 - t0) * y - 2 * power(sigma * w * sqrt(t1 - t0), 2) * (1 - gamma) * y + power(mu_tilde(w) * (t1 - t0), 2)) / power(sigma * w * sqrt(t1 - t0), 2)) * (1. - exp(-2. * (power(log(p1 / p0), 2) - y * log(p1 / p0)) / (power(sigma * w, 2.) * (t1 - t0))))

def psi_l(a, l, delta):
    return l + delta * maximum(alpha * a - l, 0) - maximum(l - a, 0)

def upsilon_l(t, l0, beta, d0):
    return minimum(l0 * exp(rho * t), (1 - beta) * d0 * exp(rho * t))

def u(x):
    return power(x, 1. - gamma) / (1. - gamma)

#https://stackoverflow.com/questions/59339256/matplotlib-plot-over-a-triangular-region
fac = 1e-3; k = (1 + fac) * d0; nu = 0.1; delta = 1
x = linspace(1e-4, T-1e-4, 150)
def _(x):
    return f(0, x, a0, k, w1)
ff = frompyfunc(_, 1, 1)
#plt.plot(x, ff(x))

def xyz_ff(k, w1, w2):
    x = linspace(1e-5, T, 200)
    y = linspace(1e-5, T, 200)
    X, Y = meshgrid(x, y)
    def _(tau, tauh):
        if tauh > 0 and tau > tauh:
            return u(exp(r * (T - tau)) * upsilon_l(tau, l0, beta, d0)) * f(0, tauh, a0, k, w1) * f(tauh, tau, (1 + nu) * k, d0, w2)
        else:
            return inf 
    ff = frompyfunc(_, 2, 1)
    Z = ff(X, Y)
    return X, Y, Z

def xyz_fg(k, w1, w2):
    x = linspace(log(d0 / k) + 1e-4, 0.5, 2000)
    y = linspace(0.96, T-1e-4, 2000)
    X, Y = meshgrid(x, y)
    def _(y, tauh):
        if tauh > 0 and tauh < T:
            #return u(psi_l(k * exp(rho * T + y), l0 * exp(rho * T), delta)) 
            return f(0, tauh, a0, k, w1) * gg(y, tauh, T, k, d0, w2)
        else:
            return inf
    ff = frompyfunc(_, 2, 1)
    Z = ff(X, Y)
    return X, Y, Z

def draw(a, X, Y, Z):
    a.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    a.view_init(azim=-155, elev=10)
    #a.view_init(azim=-165, elev=-168)
    a.set_xlabel(r'$y$')
    a.set_ylabel(r'$\tau$')
    a.locator_params(nbins=5, axis='x')
    a.locator_params(nbins=6, axis='y')

fig = plt.figure(figsize=plt.figaspect(0.4))
#ax = fig.add_subplot(1, 2, 1, projection='3d')
ax = fig.gca(projection='3d')
#X, Y, Z = xyz_fg(k, 0.5, 0.25)
X, Y, Z = xyz_fg(k, 0.1, 0.001)
draw(ax, X, Y, Z)
#ax = fig.add_subplot(1, 2, 2, projection='3d')
#X, Y, Z = xyz_fg(k, 0.5, 0.05)
#draw(ax, X, Y, Z)
#fig.tight_layout()
#fig.savefig('2.pdf', bbox_inches='tight', dpi=300)
