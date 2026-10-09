"""Named synthetic calibration. Levels, risk premia and facilities are inputs."""
from __future__ import annotations
import numpy as np
from .model import CurveModel,CurveSpec


def example_model(jumps=True):
    factors=('compute level','front scarcity','back slope','generation basis','regional basis','power','interest rate','funding liquidity','service basis')
    k=np.array([0.,4.,.35,.8,1.5,1.2,.5,1.,2.])
    sig=np.array([.18,.32,.14,.23,.20,.30,.18,.30,.18])
    # Cholesky root, shared compute/power/rates/funding shocks.
    A=np.eye(9);A[5,0]=.25;A[6,0]=-.15;A[7,0]=-.4;A[7,5]=.2;A[8,4]=.25
    C=A@A.T;C=C/np.sqrt(np.outer(np.diag(C),np.diag(C)))
    diffusion=np.diag(sig)@np.linalg.cholesky(C)
    dates=(0.,.25,.5,1.,2.,3.,5.,7.)
    vals=np.array([2.5,2.48,2.42,2.3,2.05,1.9,1.7,1.6])
    def spec(name,kind,values,l,unit='USD/GPU-hour',ob='reference'):
        return CurveSpec(name,kind,dates,tuple(values),tuple(l),unit,ob)
    a=np.array([1.,1.,.8,.6,0,0,0,0,0]);b=a.copy();b[3]=-.6;b[1]=.8
    curves=[spec('G1','log',vals,a,ob='benchmark'),spec('G2','log',vals*1.5,b,ob='benchmark')]
    for name,base,mult in [('G1',a,1.),('G2',b,1.5)]:
        for region,scale,r in [('US',1.,-.35),('SG',1.10,.65)]:
            l=base.copy();l[4]=r;l[8]=.30 if name=='G1' else .45
            curves.append(spec(name+'_'+region,'log',vals*mult*scale,l))
    curves += [spec('power_US','normal',np.array([70,72,74,76,78,80,82,84]),[8,5,0,0,2,45,2,0,0], 'USD/MWh','benchmark'),
               spec('power_SG','normal',np.array([110,112,115,118,120,122,124,126]),[8,6,0,0,12,42,2,0,0], 'USD/MWh'),
               spec('funding','funding',np.full(8,.055),[0,0,0,0,0,0,0,1,0],'annual spread','commercial')]
    m=np.array([-.12,-.16,-.07,.12,-.06,.15,.03,.35,-.08])
    sd=np.array([.07,.10,.04,.08,.05,.12,.02,.15,.05]);jc=np.diag(sd**2)
    return CurveModel(factors,k,diffusion,curves,physical_drift=[-.005,.04,-.015,.015,0,.02,.015,.04,0],
         jump_intensity_q=.30 if jumps else 0.,jump_intensity_p=.45 if jumps else 0.,
         jump_mean_q=m*.75,jump_mean_p=m,jump_cov_q=jc,jump_cov_p=jc*1.2,
         rate_loading=[0,0,0,0,0,0,.05,0,0],rate_dates=[0,1,3,7],rate_forwards=[.035,.037,.04,.04])
