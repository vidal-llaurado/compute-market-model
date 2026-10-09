"""Event-driven placement and checkpoint simulation.

This separately tests pooled-capacity assumptions. Each job needs one contiguous
cluster and a fixed GPU count. Dispatch uses current observations."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Cluster:
    name: str
    gpus: int
    useful_units_per_gpu_hour: float
    price_per_gpu_hour: float
    def __post_init__(self):
        if not self.name or not isinstance(self.gpus,int) or self.gpus<1:
            raise ValueError('named cluster and positive integer GPUs required')
        if not np.isfinite(self.useful_units_per_gpu_hour) or self.useful_units_per_gpu_hour<=0 or not np.isfinite(self.price_per_gpu_hour) or self.price_per_gpu_hour<0:
            raise ValueError('positive throughput and nonnegative price required')


@dataclass(frozen=True)
class Workload:
    name: str
    arrival: float
    deadline: float
    useful_units: float
    gpus: int
    eligible_clusters: tuple[str,...]
    checkpoint_hours: float=1.
    interruption_allowed: bool=True
    def __post_init__(self):
        if not self.name or not all(np.isfinite([self.arrival,self.deadline,self.useful_units,self.checkpoint_hours])):
            raise ValueError('named workload with finite values required')
        if self.arrival<0 or self.deadline<=self.arrival or self.useful_units<=0 or self.checkpoint_hours<=0:
            raise ValueError('invalid arrival/deadline/work/checkpoint interval')
        if not isinstance(self.gpus,int) or self.gpus<1 or not self.eligible_clusters:
            raise ValueError('positive integer GPUs and eligible clusters required')


@dataclass(frozen=True)
class Outage:
    cluster: str
    start: float
    end: float
    def __post_init__(self):
        if not np.isfinite(self.start) or not np.isfinite(self.end) or self.start<0 or self.end<=self.start:
            raise ValueError('finite nonnegative outage start and later end required')


def simulate_workloads(clusters, jobs, outages=(), horizon=None):
    """Earliest-deadline, cheapest-useful-work dispatch on currently available clusters.

    Tie convention: completion/checkpoint, then outage/deadline, then dispatch.
    Outages roll uncheckpointed progress back; incurred GPU-hour costs remain.
    """
    names=[x.name for x in clusters];jobnames=[x.name for x in jobs]
    if len(set(names))!=len(names) or len(set(jobnames))!=len(jobnames) or not clusters:
        raise ValueError('nonempty unique clusters and unique workloads required')
    if any(set(j.eligible_clusters)-set(names) for j in jobs) or any(o.cluster not in names for o in outages):
        raise ValueError('unknown cluster')
    if not jobs:return {'jobs':[],'completed':0,'deadline_failures':0,'total_cost':0.,'max_gpu_overallocation':0.}
    H=max(j.deadline for j in jobs) if horizon is None else float(horizon)
    if not np.isfinite(H) or H<=0:raise ValueError('positive finite horizon required')
    clusters={x.name:x for x in clusters}
    state={j.name:dict(job=j,status='waiting',cluster=None,work_done=0.,checkpoint=0.,
                      next_checkpoint=np.inf,cost=0.,gpu_hours=0.,lost_work=0.,interruptions=0,
                      sla_breach=False,completion=None) for j in jobs}
    time=0.;maxover=0.;history=[];eps=1e-9;iterations=0
    while time<H-eps:
        iterations+=1
        if iterations>100000:raise RuntimeError('workload event limit exceeded')
        down={o.cluster for o in outages if o.start<=time+eps and o.end>time+eps}
        for row in state.values():
            j=row['job']
            if row['status']=='running' and row['cluster'] in down:
                lost=row['work_done']-row['checkpoint'];row['lost_work']+=lost
                row['work_done']=row['checkpoint'];row['interruptions']+=1
                row['sla_breach']|=not j.interruption_allowed
                row['status']='waiting';row['cluster']=None;row['next_checkpoint']=np.inf
            if row['status'] not in ('completed','deadline_failed') and j.deadline<=time+eps:
                row['status']='deadline_failed';row['cluster']=None
        used={k:sum(r['job'].gpus for r in state.values() if r['status']=='running' and r['cluster']==k) for k in names}
        pending=sorted((r for r in state.values() if r['status']=='waiting' and r['job'].arrival<=time+eps),key=lambda r:(r['job'].deadline,r['job'].name))
        for row in pending:
            j=row['job']
            eligible=[k for k in j.eligible_clusters if k not in down and clusters[k].gpus-used[k]>=j.gpus]
            if eligible:
                k=min(eligible,key=lambda k:(clusters[k].price_per_gpu_hour/clusters[k].useful_units_per_gpu_hour,k))
                row.update(status='running',cluster=k,next_checkpoint=time+j.checkpoint_hours)
                used[k]+=j.gpus
        maxover=max(maxover,max((used[k]-clusters[k].gpus for k in names),default=0))
        future=[H]
        for row in state.values():
            j=row['job']
            if row['status'] in ('completed','deadline_failed'):continue
            if j.arrival>time+eps:future.append(j.arrival)
            if j.deadline>time+eps:future.append(j.deadline)
            if row['status']=='running':
                speed=j.gpus*clusters[row['cluster']].useful_units_per_gpu_hour
                future.extend([time+(j.useful_units-row['work_done'])/speed,row['next_checkpoint']])
        for o in outages:
            future.extend(t for t in (o.start,o.end) if t>time+eps)
        nxt=min(t for t in future if t>time+eps);nxt=min(nxt,H);dt=nxt-time
        for row in state.values():
            if row['status']=='running':
                j=row['job'];cluster=clusters[row['cluster']]
                row['work_done']+=dt*j.gpus*cluster.useful_units_per_gpu_hour
                row['cost']+=dt*j.gpus*cluster.price_per_gpu_hour;row['gpu_hours']+=dt*j.gpus
                if row['work_done']>=j.useful_units-eps:
                    row['status']='completed';row['completion']=nxt;row['cluster']=None
                    row['work_done']=j.useful_units
                elif row['next_checkpoint']<=nxt+eps:
                    row['checkpoint']=row['work_done'];row['next_checkpoint']=nxt+j.checkpoint_hours
        history.append({'start':time,'end':nxt,'allocated_gpus':dict(used)})
        time=nxt
    records=[]
    for row in state.values():
        if row['status']!='completed' and row['job'].deadline<=H+eps:
            row['status']='deadline_failed'
        records.append({k:v for k,v in row.items() if k not in ('job','cluster','next_checkpoint')}|{'name':row['job'].name})
    return {'jobs':records,'completed':sum(r['status']=='completed' for r in records),
            'deadline_failures':sum(r['status']=='deadline_failed' for r in records),
            'total_cost':sum(r['cost'] for r in records),'max_gpu_overallocation':maxover,
            'history':history,'scope':'one-cluster fixed-concurrency jobs; no financial dispatch integration'}
