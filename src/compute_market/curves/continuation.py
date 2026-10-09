"""Fit curve-state continuations for a specified operating plan.

Independent training paths estimate future components. Current dispatch and
financing remain LP decisions; the continuation approximates a plan's value."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .portfolio import spot_state,hardware_marks


def features(x,scale):
    z=np.atleast_2d(np.asarray(x,float))/scale;k=z.shape[1]
    return np.column_stack([np.ones(len(z)),z]+[z[:,i]*z[:,j] for i in range(k) for j in range(i,k)])


@dataclass
class ContinuationSurface:
    times: np.ndarray
    coefficients: np.ndarray
    scales: np.ndarray
    training_paths: int
    model_sha256: str
    diagnostics: dict
    def components(self,n,x):return features(x,self.scales[n])@self.coefficients[n]
    def value(self,n,x,holdings,debt,term,c,customer_alive=True):
        Y=self.components(n,x);h=np.asarray(holdings);f=np.asarray(c.firm_blocks);owned=np.minimum(h,f);spare=np.maximum(h-f,0);outside=np.maximum(f-h,0)
        # Terminal equipment proceeds are kept nonnegative; price curves are never clipped.
        hardware=np.maximum(Y[:,16:20],0)
        v=Y[:,:4]@spare-Y[:,4:8]@owned-Y[:,8:12]@outside+Y[:,12:16]@(f*customer_alive)+hardware@h
        remaining=max(0,len(self.times)-1-n)
        return v-debt*(1+Y[:,20])-term*(1+remaining*c.fixed_term_coupon*c.year_fraction)-remaining*c.fixed_overhead_per_step
    def save(self,path):
        import json
        np.savez_compressed(path,times=self.times,coefficients=self.coefficients,scales=self.scales,
            metadata=np.array(json.dumps({'training_paths':self.training_paths,'model_sha256':self.model_sha256,'diagnostics':self.diagnostics})))
    @classmethod
    def load(cls,path):
        import json
        with np.load(path,allow_pickle=False) as d:return cls(d['times'],d['coefficients'],d['scales'],**json.loads(str(d['metadata'])))


def train_continuation(case,tape,ridge=1e-4):
    m=case.model;c=case.fleet;X=tape.curve.factors;dates=tape.curve.times;N=len(dates);p=len(X);k=m.dimension
    gh=np.asarray(c.gpus_per_block)*c.hours_per_step;Y=np.zeros((p,N,21));terminal=tape.hardware[:,-1]*(c.liquidation_fraction-c.liquidation_impact)
    Y[:,-1,16:20]=terminal
    for n in range(N-2,-1,-1):
        Y[:,n]=Y[:,n+1];local,_,cost=spot_state(m,dates[n],X[:,n],c)
        a=tape.availability[:,n]
        Y[:,n,:4]+=np.maximum(local-cost,0)*gh*c.expected_spot_utilization*a
        Y[:,n,4:8]+=(a*cost+(1-a)*c.replacement_multiplier*local)*gh
        Y[:,n,8:12]+=c.replacement_multiplier*local*gh
        Y[:,n,12:16]+=np.asarray(c.firm_rates)*gh*tape.customer_alive[:,n,None]
        Y[:,n,20]+=c.year_fraction*(m.short_rate(dates[n],X[:,n])+m.value('funding',dates[n],X[:,n],dates[n]))
    nf=1+k+k*(k+1)//2;coef=np.zeros((N,nf,21));scales=np.maximum(X.std(axis=0),.03)
    rmse=[]
    for n in range(N):
        Z=features(X[:,n],scales[n]);pen=np.eye(nf)*ridge;pen[0,0]=0
        coef[n]=np.linalg.solve(Z.T@Z+pen+np.eye(nf)*1e-12,Z.T@Y[:,n])
        rmse.append(np.sqrt(np.mean((Z@coef[n]-Y[:,n])**2,axis=0)).tolist())
    return ContinuationSurface(dates,coef,scales,p,m.fingerprint,{'ridge':ridge,'features':'quadratic factors with cross terms','component_training_rmse':rmse,'targets':'specified per-block operating plan and terminal appraisal; not clairvoyant control','training_seed':tape.curve.seed})
