#!/usr/bin/env python3
"""Compare solver certificates with and without presolve.

Across installations, compare fingerprints, feasibility and primary objectives;
tied optimal policies may differ.
"""
from __future__ import annotations
from pathlib import Path
import argparse,json,platform,sys
import numpy as np
import scipy
from scipy.optimize import linprog,milp,Bounds,LinearConstraint
from scipy.sparse import vstack
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from compute_market.case import ValuationCase
from compute_market.dynamic_hedging import compile_dynamic,compiled_fingerprint,solver_certificate
from compute_market.hedge_audit import audit_dynamic


def check(case):
    rows=[]
    for mode in ('unhedged','static','dynamic'):
        lp,v,obj,c,p=compile_dynamic(case.portfolio,case.market,case.book,mode=mode)
        ae,au=lp.matrix(lp.eq),lp.matrix(lp.ub);be=np.asarray(lp.eq_rhs);bu=np.asarray(lp.ub_rhs)
        for presolve in (True,False):
            if any(lp.integral):
                result=milp(obj,integrality=lp.integral,bounds=Bounds(lp.lower,lp.upper),
                    constraints=LinearConstraint(vstack([ae,au]).tocsc(),np.r_[be,np.full(len(bu),-np.inf)],np.r_[be,bu]),
                    options={'presolve':presolve,'mip_rel_gap':1e-8,'time_limit':120.})
            else:
                result=linprog(obj,A_eq=ae,b_eq=be,A_ub=au,b_ub=bu,bounds=list(zip(lp.lower,lp.upper)),
                    method='highs',options={'presolve':presolve,'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8,'time_limit':120.})
            row={'mode':mode,'presolve':presolve,'case_sha256':case.fingerprint,
                 'compiled_sha256':compiled_fingerprint(lp,obj),'compiled_integer_variables':sum(lp.integral),
                 'certificate':solver_certificate(result,'minimum nested loss',lp,obj)}
            if result.success:
                policy={name:result.x[ids].tolist() for name,ids in v.items()}
                row['independent_audit']=audit_dynamic(p,case.market,c,policy,0.)
                row['equality_residual']=float(np.max(np.abs(ae@result.x-be),initial=0.))
                row['inequality_violation']=float(np.max(au@result.x-bu,initial=0.))
            rows.append(row)
    return {'project':'Compute Market','case':case.name,'environment':{'python':platform.python_version(),
        'numpy':np.__version__,'scipy':scipy.__version__},'runs':rows}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('case',type=Path)
    parser.add_argument('--output',type=Path,default=Path('results/validation/solver_checks.json'));a=parser.parse_args()
    result=check(ValuationCase.load(a.case));a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False));print(a.output)
