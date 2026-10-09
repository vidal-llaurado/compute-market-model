"""Whole-request, same-cluster, multi-interval placement using integer optimization."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class ClusterRequest:
    name: str
    gpus: int
    start: int
    end: int                         # exclusive
    eligible_clusters: tuple[int,...]
    contribution: float
    mandatory: bool = True


def allocate(capacity, requests: list[ClusterRequest], time_limit: float=10.) -> dict:
    """capacity[interval, physical cluster] is the remaining GPU count.

    One binary variable chooses a single cluster for the entire interval range.
    No pooling of disconnected spare GPUs and no migration are allowed.
    """
    cap=np.asarray(capacity,float)
    if cap.ndim != 2 or not cap.size or not np.isfinite(cap).all() or np.any(cap < 0) or not np.allclose(cap,np.rint(cap)):
        raise ValueError("nonnegative integer capacity matrix required")
    if len({r.name for r in requests}) != len(requests): raise ValueError("unique request names required")
    pairs=[]
    for i,r in enumerate(requests):
        if not isinstance(r.gpus,int) or r.gpus <= 0 or not 0 <= r.start < r.end <= cap.shape[0] or not np.isfinite(r.contribution):
            raise ValueError("invalid request dimensions")
        if len(set(r.eligible_clusters)) != len(r.eligible_clusters) or any(not 0 <= c < cap.shape[1] for c in r.eligible_clusters):
            raise ValueError("invalid cluster eligibility")
        if r.mandatory and not r.eligible_clusters:
            return {"status":"infeasible","assignments":{},"value":None}
        pairs.extend((i,c) for c in r.eligible_clusters)
    if not pairs:
        return {"status":"optimal","assignments":{},"value":0.}
    rows,cols,data=[],[],[]; lower=[]; upper=[]
    for t in range(cap.shape[0]):
        for cluster in range(cap.shape[1]):
            row=len(upper);lower.append(-np.inf);upper.append(cap[t,cluster])
            for k,(i,c) in enumerate(pairs):
                r=requests[i]
                if c == cluster and r.start <= t < r.end:
                    rows.append(row);cols.append(k);data.append(r.gpus)
    for i,r in enumerate(requests):
        row=len(upper);lower.append(1. if r.mandatory else 0.);upper.append(1.)
        for k,(idx,c) in enumerate(pairs):
            if idx == i: rows.append(row);cols.append(k);data.append(1.)
    a=coo_matrix((data,(rows,cols)),shape=(len(upper),len(pairs))).tocsc()
    objective=-np.array([requests[i].contribution for i,c in pairs])
    result=milp(objective,integrality=np.ones(len(pairs)),bounds=Bounds(0,1),
                constraints=LinearConstraint(a,lower,upper),options={"time_limit":time_limit,"mip_rel_gap":1e-10})
    if result.status == 2: return {"status":"infeasible","assignments":{},"value":None}
    if not result.success: raise RuntimeError(f"allocation solver failed: {result.message}")
    assignments={requests[i].name:c for k,(i,c) in enumerate(pairs) if result.x[k]>.5}
    return {"status":"optimal","assignments":assignments,"value":float(-result.fun)}
