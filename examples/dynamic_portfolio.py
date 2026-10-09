"""Price the offered compute reservation from the shared curve case."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from compute_market.curves.portfolio import CurveCase
from compute_market.curves.adapter import finite_case
from compute_market.curves.study import write
if __name__=='__main__':
    curve_case=CurveCase.load(ROOT/'examples/curve_case.json')
    sampled=finite_case(curve_case,branches=8,periods=2,seed=611)
    result=sampled.quote(price_tolerance=.05)
    out=ROOT/'work/example';out.mkdir(parents=True,exist_ok=True)
    sampled.save(out/'finite_case.json');write(out/'quote.json',result)
    print(json.dumps({'reservation':result['reservation'],'incremental_hedge':result['root_incremental_hedge']},indent=2))
