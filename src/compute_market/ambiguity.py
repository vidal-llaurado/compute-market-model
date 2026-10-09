"""Rectangular total-variation ambiguity for nested mean/expected shortfall.

The support LP and its dual solve the supplied finite-support problem
independently. The TV radius controls sensitivity; it is not a confidence level."""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
from .risk import distribution


def envelope(probabilities, alpha: float, radius: float):
    """Return Ax<=b, Ex=d for x=(p,q,u)>=0, |p-p0|<=u.

    q sums to 1 and q_i <= p_i/(1-alpha). TV(p,p0)<=radius.
    """
    _, p0 = distribution(np.zeros(len(probabilities)), probabilities)
    if not 0 <= alpha < 1 or not 0 <= radius <= 1:
        raise ValueError('alpha in [0,1), TV radius in [0,1] required')
    m = len(p0)
    A = np.zeros((3*m+1, 3*m)); b = np.zeros(3*m+1)
    E = np.zeros((2, 3*m)); E[0,:m]=1.; E[1,m:2*m]=1.
    for i in range(m):
        A[i,i] = -1/(1-alpha); A[i,m+i]=1.
        A[m+i,i]=1.; A[m+i,2*m+i]=-1.; b[m+i]=p0[i]
        A[2*m+i,i]=-1.; A[2*m+i,2*m+i]=-1.; b[2*m+i]=-p0[i]
    A[-1,2*m:] = 1.; b[-1] = 2*radius
    return A,b,E,np.ones(2)


def robust_risk(losses, probabilities, alpha=.95, weight=.25, radius=0.):
    """Exact worst conditional risk and an optimizing probability distortion."""
    losses, p0 = distribution(losses, probabilities)
    if not 0 <= weight <= 1:
        raise ValueError('risk weight in [0,1] required')
    A,b,E,d=envelope(p0,alpha,radius); m=len(p0)
    c=np.r_[(1-weight)*losses,weight*losses,np.zeros(m)]
    fit=linprog(-c,A_ub=A,b_ub=b,A_eq=E,b_eq=d,bounds=(0,None),method='highs')
    if not fit.success:
        raise RuntimeError(f'robust risk support solve failed: {fit.message}')
    p,q=fit.x[:m],fit.x[m:2*m]
    return {'risk':float(-fit.fun),'probabilities':p.tolist(),
            'tail_probabilities':q.tolist(),
            'effective_weights':((1-weight)*p+weight*q).tolist(),
            'tv_distance':float(.5*np.abs(p-p0).sum())}


def add_risk_epigraph(lp, loss_variable, child_variables, probabilities,
                      discounts, alpha, weight, radius, label):
    """Dualize support LP inside a parent LP/MILP; returns no extra primal decisions."""
    A,b,E,d=envelope(probabilities,alpha,radius)
    m=len(probabilities)
    z=[lp.var(f'{label}/ambiguity_dual[{i}]') for i in range(len(b))]
    y=[lp.var(f'{label}/ambiguity_eq_dual[{i}]',-np.inf) for i in range(len(d))]
    for k in range(3*m):
        row={z[i]:-A[i,k] for i in range(len(b)) if A[i,k]}
        row.update({y[i]:-E[i,k] for i in range(len(d)) if E[i,k]})
        if k < m:
            row[child_variables[k]]=(1-weight)*discounts[k]
        elif k < 2*m:
            row[child_variables[k-m]]=weight*discounts[k-m]
        lp.inequality(row,0.,f'{label}/ambiguity_feasibility[{k}]')
    row={loss_variable:-1.}
    row.update({z[i]:float(b[i]) for i in range(len(b)) if b[i]})
    row.update({y[i]:float(d[i]) for i in range(len(d)) if d[i]})
    lp.inequality(row,0.,f'{label}/ambiguity_risk')


def robust_nested_value(tree, leaf_values, alpha, weight, discounts=None, radius=0.):
    value=np.array(leaf_values,float,copy=True)
    discount=np.ones(tree.size) if discounts is None else np.asarray(discounts,float)
    radii=np.full(tree.size,float(radius)) if np.ndim(radius)==0 else np.asarray(radius,float)
    if value.shape != (tree.size,) or discount.shape != value.shape or radii.shape != value.shape:
        raise ValueError('full node vectors required')
    if not np.isfinite(value).all() or not np.isfinite(discount).all() or np.any(discount<=0) or not np.isfinite(radii).all():
        raise ValueError('finite values, radii and positive discounts required')
    if np.any((radii<0)|(radii>1)):
        raise ValueError('TV radii must be in [0,1]')
    for n in reversed(range(tree.size)):
        ch=tree.children[n]
        if ch:
            answer=robust_risk(-discount[ch]*value[ch],
                              [tree.probability[k] for k in ch],alpha,weight,radii[n])
            value[n]=-answer['risk']
    return value
