"""Scratch validation of a coalition-span/Shapley penalty-minimization theorem by LP."""
import itertools
import math
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack


def run(n, epsilon):
    keys = list(itertools.product((-1, 0, 1), repeat=n))
    index = {key: j for j, key in enumerate(keys)}
    profiles = list(itertools.product((0, 1), repeat=n))
    nv = len(keys)
    full = (1 << n) - 1

    def value(a, mask):
        vec = np.zeros(nv)
        vec[index[tuple(a[i] if mask & (1 << i) else -1 for i in range(n))]] = 1
        return vec

    def marginal(a, i, mask):
        return value(a, mask | (1 << i)) - value(a, mask)

    def penalty(a, i):
        vec = np.zeros(nv)
        for mask in range(1 << n):
            if mask & (1 << i):
                continue
            k = mask.bit_count()
            weight = math.factorial(k) * math.factorial(n-k-1) / math.factorial(n)
            vec += weight * marginal(a, i, mask)
        return vec

    def potential(a):
        vec = np.zeros(nv)
        for mask in range(1, 1 << n):
            k = mask.bit_count()
            weight = math.factorial(k-1) * math.factorial(n-k) / math.factorial(n)
            vec += weight * (value(a, mask) - value(a, 0))
        return vec

    rows = []
    for a in profiles:
        for i in range(n):
            ds = [marginal(a, i, s) for s in range(1 << n) if not s & (1 << i)]
            rows.extend(x-y for j, x in enumerate(ds) for k, y in enumerate(ds) if j != k)
    constraints = csr_matrix(np.array(rows))
    rhs = np.full(len(rows), epsilon)
    bounds = [(0, 1)] * nv
    bounds[index[(-1,) * n]] = (0.8, 0.8)
    zero, one = (0,) * n, (1,) * n
    deviations = []
    for i in range(n):
        alt = tuple(1 if k == i else 0 for k in range(n))
        deviations.append(penalty(zero, i) - penalty(alt, i))
    ne_constraints = vstack([constraints, csr_matrix(deviations)])
    ne_rhs = np.concatenate([rhs, np.zeros(n)])

    lossgap_obj = value(zero, full) - value(one, full)
    res = linprog(-lossgap_obj, A_ub=ne_constraints, b_ub=ne_rhs,
                  bounds=bounds, method='highs')
    if not res.success:
        raise RuntimeError(res.message)
    max_gap = lossgap_obj @ res.x

    other = (0,) + (1,) * (n-1)
    env_obj = penalty(other, 0) - penalty(zero, 0)
    env = linprog(-env_obj, A_ub=constraints, b_ub=rhs,
                  bounds=bounds, method='highs')
    if not env.success:
        raise RuntimeError(env.message)

    pot_residual = 0.0
    efficiency_residual = 0.0
    for a in profiles:
        efficiency_residual = max(efficiency_residual, abs(
            (sum((penalty(a, i) for i in range(n)), start=np.zeros(nv))
             - value(a, full) + value(a, 0)) @ res.x))
        for i in range(n):
            alt = tuple(1-a[k] if k == i else a[k] for k in range(n))
            residual = (potential(alt)-potential(a)-penalty(alt, i)+penalty(a, i)) @ res.x
            pot_residual = max(pot_residual, abs(residual))

    assert max_gap <= n * epsilon + 1e-8
    assert env_obj @ env.x <= epsilon + 1e-8
    assert pot_residual < 1e-8
    assert efficiency_residual < 1e-8
    print(dict(n=n, epsilon=epsilon, variables=nv,
               max_equilibrium_gap=round(float(max_gap), 10),
               claimed_bound=round(n*epsilon, 10),
               max_same_mode_penalty_change=round(float(env_obj @ env.x), 10),
               potential_residual=float(pot_residual),
               efficiency_residual=float(efficiency_residual)))


for n in (2, 3, 4):
    run(n, 0.02)
run(4, 0.0)
