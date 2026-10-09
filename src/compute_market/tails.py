"""Weighted tail diagnostics for discrete scenario supports and path samples."""
from __future__ import annotations
import numpy as np


def tail_weights(losses, probabilities, alpha):
    """Normalized ES tail weights, with fractional probability at the quantile.

    Ties at the cutoff share tail mass in proportion to their probabilities.
    This avoids arbitrary effective-sample differences caused by sort order.
    """
    x=np.asarray(losses,float);p=np.asarray(probabilities,float)
    if x.ndim!=1 or not len(x) or p.shape!=x.shape or not np.isfinite(x).all() or not np.isfinite(p).all():
        raise ValueError('finite one-dimensional losses and matching probabilities required')
    if np.any(p<0) or not np.isclose(p.sum(),1.,atol=1e-10) or not 0<=alpha<1:
        raise ValueError('probabilities must sum to one and alpha must be in [0,1)')
    q=np.zeros(len(x));remaining=1-alpha
    for value in np.unique(x)[::-1]:
        ids=np.flatnonzero(x==value);mass=p[ids].sum()
        if mass<=0:continue
        taken=min(remaining,mass)
        q[ids]=p[ids]/mass*taken/(1-alpha)
        remaining-=taken
        if remaining<=1e-14:break
    return q


def tail_diagnostics(losses, probabilities=None, alpha=.95, sample_kind='finite_support'):
    x=np.asarray(losses,float)
    p=np.full(len(x),1/len(x)) if probabilities is None else np.asarray(probabilities,float)
    q=tail_weights(x,p,alpha);es=float(q@x);active=q>1e-12
    worst=float(x[p>0].max());worst_mass=float(p[(x==worst)].sum())
    return {'alpha':float(alpha),'support_size':len(x),'positive_probability_atoms':int(np.sum(p>0)),
        'tail_active_atoms':int(active.sum()),'tail_weight_ess':float(1/(q@q)),
        'tail_probability_mass':float(1-alpha),'expected_shortfall':es,
        'worst_loss':worst,'worst_atom_probability':worst_mass,
        'saturated_at_worst':bool(worst_mass>=1-alpha-1e-12),
        'sample_kind':sample_kind,
        'nominal_tail_observations':float(len(x)*(1-alpha)) if sample_kind=='iid_paths' else None,
        'tail_weights':q.tolist()}


def tree_tail_diagnostics(tree,risk_losses,alpha):
    values=np.asarray(risk_losses,float)
    if values.shape!=(tree.size,):raise ValueError('one loss value per tree node required')
    nodes=[]
    for n,children in enumerate(tree.children):
        if children:
            diag=tail_diagnostics(values[children],np.asarray(tree.probability)[children],alpha)
            diag.update(node=n,stage=tree.stage[n]);nodes.append(diag)
    return {'conditional_nodes':nodes,
        'saturated_fraction':float(np.mean([a['saturated_at_worst'] for a in nodes])) if nodes else 0.,
        'minimum_conditional_tail_ess':min([a['tail_weight_ess'] for a in nodes],default=0.),
        'mean_conditional_tail_ess':float(np.mean([a['tail_weight_ess'] for a in nodes])) if nodes else 0.}
