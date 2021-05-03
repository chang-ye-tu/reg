import os
os.chdir(os.path.dirname(__file__))
import reg
from tabulate import tabulate
from time import time
import pickle

import numpy as np
ff = '{:.6f}'.format
np.set_printoptions(formatter={'float_kind': ff})

def create_d0():
    all_d = []
    t0 = time()
    for i in range(80, 90):
        reg.d0 = i
        d = {}
        d['d0'] = reg.d0
        d['r0'] = reg.max_0()
        d['r1'] = reg.max_1()
        d['r2'] = reg.max_2()
        d['r3'] = reg.max_3()
        all_d.append(d)
    
    #with open('all_d.pkl', 'wb') as output:
    #    pickle.dump(all_d, output)

    # success, status, message, x, fun, jac, hess, nit, nfev, njev, nhev, hess_inv
    table = []
    l0 = reg.l0
    for d in all_d:
        d0, r0, r1, r2, r3 = d['d0'], d['r0'], d['r1'], d['r2'], d['r3']
        if r0['success'] and r1['success'] and r2['success'] and r3['success']:
            x0, x1, x2, x3 = r0['x'], r1['x'], r2['x'], r3['x']
            f0, f1, f2, f3 = -r0['fun'], -r1['fun'], -r2['fun'], -r3['fun']
            values = [f0, f1, f2, f3]
            index_min = min(range(len(values)), key=values.__getitem__)
            table.append([d0, x0, ff(f0/l0), '---'])
            table.append([d0, x1, ff(f1/l0), ff(x1[-1])])
            table.append([d0, x2, ff(f2/l0), ff(x2[-1])])
            table.append([d0, x3, ff(f3/l0), ff(x3[-1])])
            table.append(['zzzzz', '', '', ''])
        else:
            print('Error occured at d0 = ', d0)
            if not r0['success']:
                print('case 0 error')
            if not r1['success']:
                print('case 1 error')
            if not r2['success']:
                print('case 2 error')
            if not r3['success']:
                print('case 3 error')

    s = tabulate(table, 
        headers=['$d_0$', r'$\Theta$', r'$\ce / \mathcal{L}$', '$k_0^\star$'], 
        tablefmt='latex_raw')
    sl = []
    for ss in s.split('\n'):
        if ss.find('zzzzz') != -1:
            sl.append(r'\midrule')
        else:
            sl.append(ss.strip())
    open('tbl_d0.txt', 'w').write('\n'.join(sl))

    print('\ntime: ', time() - t0)

def create_table(d, fn):
    table = []
    for key in sorted(d):
        _ = d[key]
        theta0, theta1, theta2, theta3 = _[0], _[1], _[2], _[3]
        reg.d0, reg.beta = key
        d0 = reg.d0
        beta = reg.beta
        ce = reg.ce(reg.util_0(*theta0))
        l0 = reg.l0
        table.append([d0, beta, theta0, ff(l0), ff(ce), ff(ce/l0), ff(reg.pd(reg.prob_0(*theta0)))])
        
        ce = reg.ce(reg.util_1(*theta1))
        l0 = reg.l0
        table.append([d0, beta, theta1, ff(l0), ff(ce), ff(ce/l0), ff(reg.pd(reg.prob_1(*theta1)))])
        
        ce = reg.ce(reg.util_2(*theta2))
        l0 = reg.l0 + reg.vartheta0(*theta2)
        table.append([d0, beta, theta2, ff(l0), ff(ce), ff(ce/l0), ff(reg.pd(reg.prob_2(*theta2)))])
        
        if theta3 is not None:
            ce = reg.ce(reg.util_3(*theta3))
            l0 = reg.l0 + reg.vartheta0_(*theta3)
            table.append([d0, beta, theta3, ff(l0), ff(ce), ff(ce/l0), ff(reg.pd(reg.prob_3(*theta3)))])

        table.append(['zzzzz', '', '', '', '', '', ''])
    
    s = tabulate(table, 
                 headers=['$d_0$', r'$\beta$', r'$\Theta$', r'$\mathcal{L}$', 
                          r'$\ce$', r'$\ce/\mathcal{L}$', r'$\text{PD}$'], 
                 tablefmt='latex_raw')
    sl = []
    for ss in s.split('\n'):
        if ss.find('zzzzz') != -1:
            sl.append(r'\midrule')
        else:
            sl.append(ss.strip())
    open('%s.txt' % fn, 'w').write('\n'.join(sl))

def create_best():
    tic = time()
    r0 = reg.max_0_shgo()
    print(r0)
    print('\ntime: ', time() - tic)
    if not r0['success']:
        print('best case 0 error. exit.') 
        return

    tic = time()
    r1 = reg.max_1_shgo()
    print(r1)
    print('\ntime: ', time() - tic)
    if not r1['success']:
        print('best case 1 error. exit.') 
        return

    tic = time()
    r2 = reg.max_2_shgo()
    print(r2)
    print('\ntime: ', time() - tic)
    if not r2['success']:
        print('best case 2 error. exit.')
        return

    tic = time()
    r3 = reg.max_3_shgo()
    print(r3)
    print('\ntime: ', time() - tic)
    if not r3['success']:
        print('best case 3 error. exit.')
        return

    x0, x1, x2, x3 = r0['x'], r1['x'], r2['x'], r3['x']
    table = []

    ce = reg.ce(reg.util_0(*x0))
    l0 = reg.l0
    table.append([0, x0[:2], '---', ff(x0[2]), ff(l0), ff(ce/l0), ff(reg.pd(reg.prob_0(*x0)))])
    table.append(['zzzzz', '', '', '', '', '', ''])         

    ce = reg.ce(reg.util_1(*x1))
    l0 = reg.l0
    table.append([1, x1[:3], ff(x1[3]), ff(x1[4]), ff(l0), ff(ce/l0), ff(reg.pd(reg.prob_1(*x1)))])
    table.append(['zzzzz', '', '', '', '', '', ''])            
    
    ce = reg.ce(reg.util_2(*x2))
    l0 = reg.l0 + reg.vartheta0(*x2)
    table.append([2, x2[:3], ff(x2[3]), ff(x2[4]), ff(l0), ff(ce/l0), ff(reg.pd(reg.prob_2(*x2)))])
    table.append(['zzzzz', '', '', '', '', '', ''])            

    ce = reg.ce(reg.util_3(*x3))
    l0 = reg.l0 + reg.vartheta0_(*x3)
    table.append([2, x3[:3], ff(x3[4]), ff(x3[5]), ff(l0), ff(ce/l0), ff(reg.pd(reg.prob_3(*x3)))])

    s = tabulate(table, 
        headers=['', r'$\Theta$', '$k_0^\star$', '$d_0^\star$', 
                 r'$\mathcal{L}$', r'$\ce / \mathcal{L}$', r'$\text{PD}$'], 
        tablefmt='latex_raw')
    sl = []
    for ss in s.split('\n'):
        if ss.find('zzzzz') != -1:
            sl.append(r'\midrule')
        else:
            sl.append(ss.strip())
    open('tbl_best.txt', 'w').write('\n'.join(sl))

def create_nu():
    table = []
    for i in [10, 15, 20, 25, 30, 35, 40]: #range(1, 51):
        print('nu: ', i)
        t0 = time()
        r = reg.max_2_nu_shgo(i * 0.01)
        if r['success']:
            x, fun = r['x'], -r['fun']
            table.append([i, x, ff(fun/reg.l0)])
        else:
            print('error in i = ', i)
        print('\ntime: ', time() - t0)
        s = tabulate(table, 
            headers=['$nu$', r'$\Theta$', r'$\ce / \mathcal{L}$'], 
            tablefmt='latex_raw')
        sl = []
        for ss in s.split('\n'):
            if ss.find('zzzzz') != -1:
                sl.append(r'\midrule')
            else:
                sl.append(ss.strip())
        open('tbl_nu.txt', 'w').write('\n'.join(sl))

#reg.d0 = 90
#r3 = reg.max_1_shgo_0()
#l0 = reg.l0
#x3 = r3['x']
#f3 = -r3['fun']
#print([reg.d0, x3, ff(f3/l0), ff(x3[-1])])

#d_my = {}
#for l in ((90, 0), (90, 0.1), (94, 0), (94, 0.1)):
#    reg.d0, reg.beta = l
#    r0, r1, r2, r3 = reg.max_0(), reg.max_1(False), reg.max_2(False), reg.max_3(False)
#    d_my[l] = [r0['x'], r1['x'], r2['x'], r3['x']] 
#with open('d_my.pkl', 'wb') as output:
#    pickle.dump(d_my, output)
#with open('d_my.pkl', 'rb') as output:
#    d_my = pickle.load(output)

#create_table(d_my, 'tbl_my')

#d_chen = {
#    (90, 0): [[0.141, 0.83], [0.237, 0.068, 0.745], [0.286, 0.158, 0.975], None],
#    (90, 0.1): [[0.115, 0.867], [0.231, 0.038, 0.727], [0.241, 0.143, 0.975], None], 
#    (94, 0): [[0.096, 0.86], [0.181, 0.024, 0.839], [0.267, 0.186, 1.], None],
#    (94, 0.1): [[0.072, 0.937], [0.179, 0.02, 0.844], [0.247, 0.173, 1.], None]
#}
#create_table(d_chen, 'tbl_chen')

#create_best()

#with open('all_d.pkl', 'rb') as output:
#    all_d = pickle.load(output)
