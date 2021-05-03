from numpy import maximum, minimum, log, exp, power, sqrt, pi, inf
from scipy.optimize import minimize, shgo
from scipy.stats import norm
from scipy.integrate import quad, nquad
from numba import float64, jit

T, gamma, sigma, mu, r, rho = 10, 3, 0.2, 0.06, 0.025, 0.02
a0, k0, d0, beta = 100., 95, 94, 0.
alpha = 0.95; l0 = alpha * a0 
epsilon = 1 - power(1 - 0.005, T)
fac, thr, bign = 1e-4, 1e-2, 1e8

@jit(float64(float64), nopython=True, cache=True)
def mu_tilde(w):
    return r + w * (mu - r) - rho - power(sigma * w, 2) / 2 

@jit(float64(float64), nopython=True, cache=True)
def mu_tilde_q(w):
    return r - rho - power(sigma * w, 2) / 2 

@jit(float64(float64, float64, float64, float64, float64), nopython=True, cache=True)
def f(t0, t1, p0, p1, w):
    return -log(p1 / p0) / (sigma * w * power(t1 - t0, 3. / 2)) / sqrt(2 * pi) * exp(-1 / 2 * ((log(p1 / p0) - mu_tilde(w) * (t1 - t0)) / (sigma * w * sqrt(t1 - t0))) ** 2)

@jit(float64(float64, float64, float64, float64, float64), nopython=True, cache=True)
def f_q(t0, t1, p0, p1, w):
    return -log(p1 / p0) / (sigma * w * power(t1 - t0, 3. / 2)) / sqrt(2 * pi) * exp(-1 / 2 * ((log(p1 / p0) - mu_tilde_q(w) * (t1 - t0)) / (sigma * w * sqrt(t1 - t0))) ** 2)

@jit(float64(float64, float64, float64, float64, float64, float64), nopython=True, cache=True)
def g(y, t0, t1, p0, p1, w):
    return 1. / (sigma * w * sqrt(t1 - t0)) / sqrt(2 * pi) * exp(-1 / 2 * power(((y - mu_tilde(w) * (t1 - t0)) / (sigma * w * sqrt(t1 - t0))), 2)) * (1. - exp(-2. * (power(log(p1 / p0), 2) - y * log(p1 / p0)) / (power(sigma * w, 2.) * (t1 - t0))))

@jit(float64(float64, float64, float64, float64, float64, float64), nopython=True, cache=True)
def gg(y, t0, t1, p0, p1, w):
    return 1. / (sigma * w * sqrt(t1 - t0)) / sqrt(2 * pi) * exp(-1. / 2 * (power(y, 2) - 2 * mu_tilde(w) * (t1 - t0) * y - 2 * power(sigma * w * sqrt(t1 - t0), 2) * (1 - gamma) * y + power(mu_tilde(w) * (t1 - t0), 2)) / power(sigma * w * sqrt(t1 - t0), 2)) * (1. - exp(-2. * (power(log(p1 / p0), 2) - y * log(p1 / p0)) / (power(sigma * w, 2.) * (t1 - t0))))

@jit(float64(float64, float64, float64, float64, float64, float64), nopython=True, cache=True)
def gg_q(y, t0, t1, p0, p1, w):
    return 1. / (sigma * w * sqrt(t1 - t0)) / sqrt(2 * pi) * exp(-1. / 2 * (power(y, 2) - 2 * mu_tilde_q(w) * (t1 - t0) * y - 2 * power(sigma * w * sqrt(t1 - t0), 2) * y + power(mu_tilde_q(w) * (t1 - t0), 2)) / power(sigma * w * sqrt(t1 - t0), 2)) * (1. - exp(-2. * (power(log(p1 / p0), 2) - y * log(p1 / p0)) / (power(sigma * w, 2.) * (t1 - t0))))

@jit(float64(float64, float64, float64), nopython=True, cache=True)
def psi_l(a, l, delta):
    return l + delta * maximum(alpha * a - l, 0) - maximum(l - a, 0)

@jit(float64(float64, float64, float64), nopython=True, cache=True)
def psi_e(a, l, delta):
    return maximum(a - l, 0) - delta * maximum(alpha * a - l, 0) 

@jit(float64(float64, float64, float64, float64), nopython=True, cache=True)
def upsilon_l(t, l0, beta, d0):
    return minimum(l0 * exp(rho * t), (1 - beta) * d0 * exp(rho * t))

@jit(float64(float64, float64, float64, float64), nopython=True, cache=True)
def upsilon_e(t, l0, beta, d0):
    return maximum((1 - beta) * d0 * exp(rho * t) - l0 * exp(rho * t),  0)

@jit(float64(float64), nopython=True, cache=True)
def u(x):
    return power(x, 1. - gamma) / (1. - gamma)

@jit(float64(float64), nopython=True, cache=True)
def ce(x):
    return power((1. - gamma) * x, 1. / (1. - gamma)) if ((1 - gamma) * x) > 0 else 0

def pd(x):
    return 1 - power(1 - x, 1. / T)

def range_tau(tauh):
    return [tauh, T]

@jit(float64(float64), nopython=True, cache=True)
def pky(w):
    return sqrt(sigma**2 * w**2 * T + (mu_tilde(w))**2 * T**2)

@jit(float64(float64), nopython=True, cache=True)
def py(w):
    return mu_tilde(w) * T + 10 * sigma * w * sqrt(T)

@jit(float64(float64), nopython=True, cache=True)
def py_q(w):
    return mu_tilde_q(w) * T + 10 * sigma * w * sqrt(T)

@jit(float64(float64, float64), nopython=True, cache=True)
def y0(w1, w2):
    return sigma**4 * w1**2 * w2**2 * gamma**2 - (2 * sigma**4 * w1**2 * w2**2 + 2 * mu_tilde(w2) * sigma**2 * w1**2) * gamma + sigma**4 * w1**2 * w2**2 + 2 * mu_tilde(w2) * sigma**2 * w1**2 + (mu_tilde(w1))**2 

@jit(float64(float64, float64, float64), nopython=True, cache=True)
def y1(w1, w2, a1):
    return (a1 * sigma**2 * w2**2 * gamma - a1 * sigma**2 * w2**2 - a1 * mu_tilde(w2)) * sqrt(y0(w1, w2)) - (3 * T * sigma**6 * w1**2 * w2**4 + (6 * T * mu_tilde(w2) * sigma**4 * w1**2 + T * (mu_tilde(w1))**2 * sigma**2) * w2**2 + 2 * T * (mu_tilde(w2))**2 * sigma**2 * w1**2) * gamma - T * sigma**6 * w1**2 * w2**4 * gamma**3 + (3 * T * sigma**6 * w1**2 * w2**4 + 3 * T * mu_tilde(w2) * sigma**4 * w1**2 * w2**2) * gamma**2 + T * sigma**6 * w1**2 * w2**4 + (3 * T * mu_tilde(w2) * sigma**4 * w1**2 + T * (mu_tilde(w1))**2 * sigma**2) * w2**2 + 2 * T * (mu_tilde(w2))**2 * sigma**2 * w1**2 + T * (mu_tilde(w1))**2 * mu_tilde(w2) 

@jit(float64(float64, float64, float64), nopython=True, cache=True)
def y_star(w1, w2, a1):
    return y1(w1, w2, a1) / y0(w1, w2)

@jit(float64(float64, float64, float64), nopython=True, cache=True)
def t_star(w1, w2, a1):
    return a1 / sqrt(y0(w1, w2))

def fair_cond_0(w, delta, d=None):
    d = d0 if d is None else d
    print('w, delta, d now: ', w, delta, d)
    ip = quad(lambda tau: exp(-r * tau) * upsilon_e(tau, l0, beta, d) * f_q(0, tau, a0, d, w), 0, T, epsabs=0, limit=200)[0] 
    print('ip: ', ip)
    
    im = quad(lambda y: exp(-r * T) * psi_e(a0 * exp(rho * T), l0 * exp(rho * T - y), delta) * gg_q(y, 0, T, a0, d, w), log(d / a0), bign, epsabs=0, points=[py_q(w), 2*py_q(w),], limit=200)[0]
    
    #im = quad(lambda y: exp(-r * T) * psi_e(a0 * exp(rho * T), l0 * exp(rho * T - y), delta) * gg_q(y, 0, T, a0, d, w), log(d / a0), inf, epsabs=0, limit=200)[0]
    print('im: ', im)
    print('ip + im: ', ip + im, '\n')

    return ip + im

def fair_cond_1(w1, w2, delta, k=None, d=None):
    k = k0 if k is None else k
    d = d0 if d is None else d
    print('w1, w2, delta, k, d now: ', w1, w2, delta, k, d)

    ipp = nquad(lambda tau, tauh: exp(-r * tau) * upsilon_e(tau, l0, beta, d) * f_q(0, tauh, a0, k, w1) * f_q(tauh, tau, k, d, w2), [range_tau, [0, T]], opts={'epsabs': 0, 'limit': 200})[0]
    print('ipp: ', ipp)

    ipm = nquad(lambda y, tauh: exp(-r * T) * psi_e(k * exp(rho * T), l0 * exp(rho * T - y), delta) * f_q(0, tauh, a0, k, w1) * gg_q(y, tauh, T, k, d, w2), [[log(d / k), bign], [0, T]], opts=[{'epsabs': 0, 'points': [py_q(w2),], 'limit': 200}, {'epsabs': 0, 'limit': 200}])[0]
    print('ipm: ', ipm)

    im = quad(lambda y: exp(-r * T) * psi_e(a0 * exp(rho * T), l0 * exp(rho * T - y), delta) * gg_q(y, 0, T, a0, k, w1), log(k / a0), bign, epsabs=0, points=[py_q(w1), 2*py_q(w1),], limit=200)[0]
    print('im: ', im)
    print('ipp + ipm + im: ', ipp + ipm + im, '\n')

    return ipp + ipm + im

def _fair_cond_2(w1, w2, nu, delta, k, d):
    ipp = nquad(lambda tau, tauh: exp(-r * tau) * upsilon_e(tau, l0, beta, d) * f_q(0, tauh, a0, k, w1) * f_q(tauh, tau, (1 + nu) * k, d, w2), [range_tau, [0, T]], opts={'epsabs': 0, 'limit': 200})[0]    
    print('ipp: ', ipp)

    ipm = nquad(lambda y, tauh: exp(-r * T) * psi_e((1 + nu) * k * exp(rho * tauh) * exp(rho * (T - tauh)), l0 * exp(rho * T - y), delta) * f_q(0, tauh, a0, k, w1) * gg_q(y, tauh, T, (1 + nu) * k, d, w2), [[log(d / ((1 + nu) * k)), bign], [0, T]], opts=[{'epsabs': 0, 'points': [py_q(w2),], 'limit': 200}, {'epsabs': 0, 'limit': 200}])[0]
    print('ipm: ', ipm)

    im = quad(lambda y: exp(-r * T) * psi_e(a0 * exp(rho * T), l0 * exp(rho * T - y), delta) * gg_q(y, 0, T, a0, k, w1), log(k / a0), bign, epsabs=0, points=[py_q(w1),], limit=200)[0]
    print('im: ', im)
    print('ipp + ipm + im: ', ipp + ipm + im, '\n')

    return ipp + ipm + im

def fair_cond_2(w, nu, delta, k=None, d=None):
    k = k0 if k is None else k
    d = d0 if d is None else d
    print('w, nu, delta, k, d now: ', w, nu, delta, k, d)
    return _fair_cond_2(w, w, nu, delta, k, d)

def fair_cond_3(w1, w2, nu, delta, k=None, d=None):
    k = k0 if k is None else k
    d = d0 if d is None else d
    print('w1, w2, nu, delta, k, d now: ', w1, w2, nu, delta, k, d)
    return _fair_cond_2(w1, w2, nu, delta, k, d)

def util_0(w, delta, d=None):
    d = d0 if d is None else d
    ip = quad(lambda tau: u(exp(r * (T - tau)) * upsilon_l(tau, l0, beta, d)) * f(0, tau, a0, d, w), 0, T, epsabs=0, limit=200)[0] 
    im = quad(lambda y: u(psi_l(a0 * exp(rho * T), l0 * exp(rho * T - y), delta)) * gg(y, 0, T, a0, d, w), log(d / a0), bign, epsabs=0, points=[py(w),], limit=200)[0]
    
    return ip + im

def util_1(w1, w2, delta, k=None, d=None):
    k = k0 if k is None else k
    d = d0 if d is None else d

    ipp = nquad(lambda tau, tauh: u(exp(r * (T - tau)) * upsilon_l(tau, l0, beta, d)) * f(0, tauh, a0, k, w1) * f(tauh, tau, k, d, w2), [range_tau, [0, T]], opts={'epsabs': 0, 'limit': 200})[0]
    
    ipm = nquad(lambda y, tauh: u(psi_l(k * exp(rho * T), l0 * exp(rho * T - y), delta)) * f(0, tauh, a0, k, w1) * gg(y, tauh, T, k, d, w2), [[log(d / k), bign], [0, T]], opts=[{'epsabs': 0, 'points': [py(w2),], 'limit': 200}, {'epsabs': 0, 'limit': 200}])[0]
    
    im = quad(lambda y: u(psi_l(a0 * exp(rho * T), l0 * exp(rho * T - y), delta)) * gg(y, 0, T, a0, k, w1), log(k / a0), bign, epsabs=0, points=[py(w1),], limit=200)[0]
    
    return ipp + ipm + im

def _util_2(w1, w2, nu, delta, k, d):
    ipp = nquad(lambda tau, tauh: u(exp(r * (T - tau)) * upsilon_l(tau, l0, beta, d)) * f(0, tauh, a0, k, w1) * f(tauh, tau, (1 + nu) * k, d, w2), [range_tau, [0, T]], opts={'epsabs': 0, 'limit': 200})[0]
    
    ipm = nquad(lambda y, tauh: u(psi_l((1 + nu) * k * exp(rho * tauh) * exp(rho * (T - tauh)), l0 * exp(rho * T - y), delta)) * f(0, tauh, a0, k, w1) * gg(y, tauh, T, (1 + nu) * k, d, w2), [[log(d / ((1 + nu) * k)), bign], [0, T]], opts=[{'epsabs': 0, 'points': [py(w2),], 'limit': 200}, {'epsabs': 0, 'limit': 200}])[0]

    im = quad(lambda y: u(psi_l(a0 * exp(rho * T), l0 * exp(rho * T - y), delta)) * gg(y, 0, T, a0, k, w1), log(k / a0), bign, epsabs=0, points=[py(w1),], limit=200)[0]

    return ipp + ipm + im

def util_2(w, nu, delta, k=None, d=None):
    k = k0 if k is None else k
    d = d0 if d is None else d
    return _util_2(w, w, nu, delta, k, d)

def util_3(w1, w2, nu, delta, k=None, d=None):
    k = k0 if k is None else k
    d = d0 if d is None else d
    return _util_2(w1, w2, nu, delta, k, d)

def _vartheta(w, nu, k):
    return quad(lambda tauh: exp(-r * tauh) * nu * k * exp(rho * tauh) * f_q(0, tauh, a0, k, w), 0, T, epsabs=0, limit=200)[0]

def vartheta0(w, nu, delta, k=None, d=None):
    k = k0 if k is None else k
    return _vartheta(w, nu, k)

def vartheta0_(w1, w2, nu, delta, k=None, d=None):
    k = k0 if k is None else k
    return _vartheta(w1, nu, k)

def prob_0(w, delta, d=None):
    d = d0 if d is None else d
    return quad(lambda tau: f(0, tau, a0, d, w), 0, T, epsabs=0, limit=200)[0]

def prob_0_exact(w, d=None):
    d = d0 if d is None else d
    return norm.cdf((log(d / a0) - mu_tilde(w) * T) / (sigma * w * sqrt(T))) + power(d / a0, 2 * mu_tilde(w) / power(sigma * w, 2)) * norm.cdf((log(d / a0) + mu_tilde(w) * T) / (sigma * w * sqrt(T)))

def prob_1(w1, w2, delta, k=None, d=None):
    k = k0 if k is None else k
    d = d0 if d is None else d
    return nquad(lambda tau, tauh: f(0, tauh, a0, k, w1) * f(tauh, tau, k, d, w2), [range_tau, [0, T]], opts={'epsabs': 0, 'limit': 200})[0]

def _prob_2(w1, w2, nu, delta, k, d):
    return nquad(lambda tau, tauh: f(0, tauh, a0, k, w1) * f(tauh, tau, (1 + nu) * k, d, w2), [range_tau, [0, T]], opts={'epsabs': 0, 'limit': 200})[0]

def prob_2(w, nu, delta, k=None, d=None):
    k = k0 if k is None else k
    d = d0 if d is None else d
    return _prob_2(w, w, nu, delta, k, d)

def prob_3(w1, w2, nu, delta, k=None, d=None):
    k = k0 if k is None else k
    d = d0 if d is None else d
    return _prob_2(w1, w2, nu, delta, k, d)

def max_0():
    bnd = ([1e-3, 1.], [0., 1])
    init = [0.5, 0.5]
    
    def cs_p(x):
        return epsilon - prob_0(*x)
    def cs_fair(x):
        return fair_cond_0(*x) - (1 - alpha) * a0
    cs = [{'type': 'ineq', 'fun': cs_p}, {'type': 'ineq', 'fun': cs_fair}]
    def aim(x):
        return -ce(util_0(*x))
    t = minimize(aim, init, bounds=bnd, constraints=cs)
    return t

def max_1(b_k=True):
    if b_k:
        bnd = ([1e-2, 1.], [1e-2, 1.], [0., 1], [d0 + thr, l0])
        init = [0.5, 0.5, 0.5, (d0 + l0 + thr) / 2.]
    else:
        bnd = ([1e-2, 1.], [1e-2, 1.], [0., 1])
        init = [0.5, 0.5, 0.5]
    
    def cs_p(x):
        return epsilon - prob_1(*x)
    def cs_fair(x):
        return fair_cond_1(*x) - (1 - alpha) * a0
    cs = [{'type': 'ineq', 'fun': cs_p}, {'type': 'ineq', 'fun': cs_fair}]
    def aim(x):
        return -ce(util_1(*x))
    t = minimize(aim, init, bounds=bnd, constraints=cs)
    return t

def max_2(b_k=True):
    if b_k:
        bnd = ([1e-2, 1], [0., 1], [0., 1], [d0 + thr, l0])
        init = [0.5, 0.5, 0.5, (d0 + l0 + thr) / 2.]
    else:
        bnd = ([1e-2, 1], [0., 1], [0., 1])
        init = [0.5, 0.5, 0.5]

    def cs_p(x):
        return epsilon - prob_2(*x)
    def cs_fair(x):
        return fair_cond_2(*x) - (1 - alpha) * a0
    cs = [{'type': 'ineq', 'fun': cs_p}, {'type': 'ineq', 'fun': cs_fair}]
    def aim(x):
        return -ce(util_2(*x)) / (1 + (vartheta0(*x) / l0))
    t = minimize(aim, init, bounds=bnd, constraints=cs)
    return t

def max_3(b_k=True):
    if b_k:
        bnd = ([1e-2, 1], [1e-2, 1], [0., 1], [0., 1], [d0 + thr, l0])
        init = [0.5, 0.5, 0.5, 0.5, (d0 + l0 + thr) / 2.]
    else:
        bnd = ([1e-2, 1], [1e-2, 1], [0., 1], [0., 1])
        init = [0.5, 0.5, 0.5, 0.5]

    def cs_p(x):
        return epsilon - prob_3(*x)
    def cs_fair(x):
        return fair_cond_3(*x) - (1 - alpha) * a0
    cs = [{'type': 'ineq', 'fun': cs_p}, {'type': 'ineq', 'fun': cs_fair}]
    def aim(x):
        return -ce(util_3(*x)) / (1 + (vartheta0_(*x) / l0))
    t = minimize(aim, init, bounds=bnd, constraints=cs)
    return t

def max_0_shgo():
    bnd = ([1e-2, 1.], [0., 1], [1e-2, l0])
    def cs_p(x):
        return epsilon - prob_0(*x)
    def cs_fair(x):
        return fair_cond_0(*x) - (1 - alpha) * a0
    cs = [{'type': 'ineq', 'fun': cs_p}, {'type': 'ineq', 'fun': cs_fair}]
    def aim(x):
        return -ce(util_0(*x))
    t = shgo(aim, bounds=bnd, constraints=cs, sampling_method='sobol')
    return t

def max_1_shgo_0():
    bnd = ([1e-2, 1.], [1e-2, 1.], [0., 1])
    
    def cs_p(x):
        return epsilon - prob_1(*x)
    def cs_fair(x):
        return fair_cond_1(*x) - (1 - alpha) * a0
    cs = [{'type': 'ineq', 'fun': cs_p}, {'type': 'ineq', 'fun': cs_fair}]
    def aim(x):
        return -ce(util_1(*x))
    t = shgo(aim, bounds=bnd, constraints=cs, sampling_method='sobol')
    return t

def max_1_shgo():
    bnd = ([1e-2, 1.], [1e-2, 1.], [0., 1], [1e-2, l0], [1e-2, l0])
    def cs_p(x):
        return epsilon - prob_1(*x)
    def cs_fair(x):
        return fair_cond_1(*x) - (1 - alpha) * a0
    cs = [{'type': 'ineq', 'fun': cs_p}, {'type': 'ineq', 'fun': cs_fair}]
    def aim(x):
        return -ce(util_1(*x))
    def cs_k(x):
        return x[3] - (1 + fac) * x[4]
    cs.append({'type': 'ineq', 'fun': cs_k})
    t = shgo(aim, bounds=bnd, constraints=cs, sampling_method='sobol')
    return t

def max_2_shgo():
    bnd = ([1e-2, 1], [0., 1], [0., 1], [1e-2, l0], [1e-2, l0])
    def cs_p(x):
        return epsilon - prob_2(*x)
    def cs_fair(x):
        return fair_cond_2(*x) - (1 - alpha) * a0
    cs = [{'type': 'ineq', 'fun': cs_p}, {'type': 'ineq', 'fun': cs_fair}]
    def aim(x):
        return -ce(util_2(*x)) / (1 + (vartheta0(*x) / l0))
    def cs_k(x):
        return x[3] - (1 + fac) * x[4]
    cs.append({'type': 'ineq', 'fun': cs_k}) 
    t = shgo(aim, bounds=bnd, constraints=cs, sampling_method='sobol')
    return t

def max_3_shgo_0():
    bnd = ([1e-2, 1], [1e-2, 1], [0., 1], [0., 1], [d0 + thr, l0])
    def cs_p(x):
        return epsilon - prob_3(*x)
    def cs_fair(x):
        return fair_cond_3(*x) - (1 - alpha) * a0
    cs = [{'type': 'ineq', 'fun': cs_p}, {'type': 'ineq', 'fun': cs_fair}]
    def aim(x):
        return -ce(util_3(*x)) / (1 + (vartheta0_(*x) / l0))
    t = shgo(aim, bounds=bnd, constraints=cs, sampling_method='sobol')
    return t

def max_3_shgo():
    bnd = ([1e-2, 1], [1e-2, 1], [0., 1], [0., 1], [1e-2, l0], [1e-2, l0])
    def cs_p(x):
        return epsilon - prob_3(*x)
    def cs_fair(x):
        return fair_cond_3(*x) - (1 - alpha) * a0
    cs = [{'type': 'ineq', 'fun': cs_p}, {'type': 'ineq', 'fun': cs_fair}]
    def aim(x):
        return -ce(util_3(*x)) / (1 + (vartheta0_(*x) / l0))
    def cs_k(x):
        return x[4] - (1 + fac) * x[5]
    cs.append({'type': 'ineq', 'fun': cs_k})
    t = shgo(aim, bounds=bnd, constraints=cs, sampling_method='sobol')
    return t
