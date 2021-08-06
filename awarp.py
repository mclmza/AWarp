import numpy as np
from numba import jit

L = 'left'
T = 'top'
D = 'diagonal'
INF = int(1e8)


@jit(nopython=True)
def ub_costs(a, b, case):
    if a > 0 and b > 0:
        return (a - b) ** 2
    elif b < 0 < a:
        if case == L:
            return a ** 2
        else:
            return -b * a ** 2
    elif b > 0 > a:
        if case == T:
            return b ** 2
        else:
            return -a * (b ** 2)
    else:
        return 0


@jit(nopython=True)
def ub_costs_constrained(a, b, mode, w, gap):
    if a > 0 and b > 0 and gap <= w:
        return (a - b) ** 2
    elif a < 0 and b < 0:
        return 0
    else:
        if mode == D:
            if b < 0 < a:
                return -b * (a**2)
            elif a < 0 < b:
                return -a * (b**2)
            else:
                return int(INF)
        elif mode == L:
            if b < 0 < a and gap <= w:
                return -b * (a**2)
            elif a < 0 < b:
                return b ** 2
            else:
                return int(INF)
        elif mode == T:
            if b < 0 < a:
                return a**2
            elif a < 0 < b and gap <= w:
                return -a * (b**2)
            else:
                return int(INF)


@jit(nopython=True)
def compute_awarp(d, x, y):
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            if i > 0 and j > 0:
                a_d = d[i, j] + ub_costs(x[i], y[j], 'diagonal')
            else:
                a_d = d[i, j] + (x[i] - y[j]) ** 2
            a_l = d[i+1, j] + ub_costs(x[i], y[j], 'top')
            a_t = d[i, j+1] + ub_costs(x[i], y[j], 'left')
            d[i+1, j+1] = min(a_d, a_t, a_l)

@jit(nopython=True)
def compute_awarp_constrained(d, x, y, w, t_x, t_y):
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            gap = np.absolute(t_x[i] - t_y[j])
            if gap > w and ((j > 0 and t_y[j-1] - t_x[i] > w) or (i > 0 and t_x[i-1] - t_y[j] > w)):
                d[i+1, j+1] = int(INF)
            else:
                if i > 0 and j > 0:
                    a_d = d[i, j] + ub_costs_constrained(x[i], y[j], D, w, gap)
                else:
                    a_d = d[i, j] + (x[i] - y[j]) ** 2
                a_l = d[i+1, j] + ub_costs_constrained(x[i], y[j], L, w, gap)
                a_t = d[i, j+1] + ub_costs_constrained(x[i], y[j], T, w, gap)

                # Avoid overflow
                val = min(a_d, a_t, a_l, INF)
                if val < 0:
                    val = int(INF)

                d[i+1, j+1] = val

def awarp(x, y, w=0):
    d = np.zeros((x.shape[0] + 1, y.shape[0] + 1)).astype(int)
    d[:, 0] = int(INF)
    d[0, :] = int(INF)
    d[0, 0] = 0

    if w > 0:
        t_x = np.zeros(x.shape[0] + 1).astype(int)
        t_y = np.zeros(y.shape[0] + 1).astype(int)

        iit = 0
        for i in range(x.shape[0]):
            if x[i] > 0:
                iit += 1
            else:
                iit += abs(x[i])
            t_x[i] = iit
        t_x[-1] = iit + 1

        iit = 0
        for i in range(y.shape[0]):
            if y[i] > 0:
                iit += 1
            else:
                iit += abs(y[i])
            t_y[i] = iit
        t_y[-1] = iit + 1
        compute_awarp_constrained(d, x, y, w, t_x, t_y)
    else:
        compute_awarp(d, x, y)

    return np.sqrt(d[-1, -1])
