"""Stochastic-curve commands and reusable finite-control numerical tools."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from .case import ValuationCase
from .optimizer import OptimizationError


def write_json(path,value):
    from .research import write
    write(Path(path),value)


def example(path,depth=3,integer_blocks=False):
    from .hedge_examples import dynamic_case
    p,m,b,c=dynamic_case(depth)
    if integer_blocks:p=p.copy(integer_blocks=True)
    case=ValuationCase('compute merchant valuation',p,m,b,c,{'data':'synthetic'})
    case.save(path)
    return case


def demo(output: Path, depth: int=3, integer_blocks: bool=False) -> dict:
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    case=example(output/'case.json',depth,integer_blocks)
    modes={mode:case.solve(mode).to_dict() for mode in ('unhedged','static','dynamic')}
    result={'project':'Compute Market','data':'synthetic','case_sha256':case.fingerprint,
            'modes':modes,'price_and_hedge':case.quote(price_tolerance=.1)}
    write_json(output/'summary.json',result)
    return result


def parser():
    p=argparse.ArgumentParser(description='Compute Market: coupled stochastic curves and financed portfolio control')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('curves',help='primary curve model, complete case, surface, pricing, study and replay')
    e=sub.add_parser('example',help='write a synthetic complete valuation case')
    e.add_argument('--output',type=Path,default=Path('examples/case.json'))
    e.add_argument('--depth',type=int,default=3,help='delivery depth of settlement template')
    e.add_argument('--integer-blocks',action='store_true')
    e.add_argument('--template',choices=['market','settlement'],default='market')
    e.add_argument('--branches',type=int,default=64)
    e.add_argument('--periods',type=int,default=2)
    e.add_argument('--seed',type=int,default=1201)
    d=sub.add_parser('demo',help='solve and price a complete synthetic example')
    d.add_argument('--output',type=Path,default=Path('results/demo'))
    d.add_argument('--depth',type=int,default=3);d.add_argument('--integer-blocks',action='store_true')
    for name in ('solve','quote','validate'):
        q=sub.add_parser(name,help=name+' one complete valuation case')
        q.add_argument('case',type=Path);q.add_argument('--output',type=Path)
        q.add_argument('--mode',choices=['unhedged','static','dynamic'],default='dynamic')
        if name=='quote':
            q.add_argument('--payment',choices=['upfront','opening_rate','closing_rate'],default='upfront')
            q.add_argument('--tolerance',type=float,default=.05)
    v=sub.add_parser('verify',help='replay a solved policy against its complete case')
    v.add_argument('case',type=Path);v.add_argument('result',type=Path)
    r=sub.add_parser('research',help='finite-kernel numerical control study; the curve study is `curves study`')
    r.add_argument('--output',type=Path,default=Path('results/study'))
    for name,default in [('paths',2048),('train-paths',4000),('stress-paths',1024),('steps',12),('scenarios',256),('seed',20260918),('bootstrap',200),('workers',4)]:
        r.add_argument('--'+name,type=int,default=default)
    r.add_argument('--quick',action='store_true')
    v=sub.add_parser('replay',help='replay the saved held-out market tape and execution mandate')
    v.add_argument('folder',type=Path)
    v.add_argument('--policy',choices=['joint_unhedged','joint_static','joint','adaptive'],default='joint')
    v.add_argument('--workers',type=int,default=1)
    v.add_argument('--paths',type=int,help='verify a declared prefix; default is the entire tape')
    return p


def main(argv=None):
    tokens=list(sys.argv[1:] if argv is None else argv)
    if tokens and tokens[0]=='curves':
        from .curves.cli import main as curve_main
        return curve_main(tokens[1:])
    args=parser().parse_args(tokens)
    try:
        if args.command=='example':
            if args.template=='market':
                from .scenarios import sampled_case
                case=sampled_case(args.branches,args.periods,args.seed)
                if args.integer_blocks:case.portfolio=case.portfolio.copy(integer_blocks=True)
                case.save(args.output)
            else:case=example(args.output,args.depth,args.integer_blocks)
            print(json.dumps({'case':str(args.output),'case_sha256':case.fingerprint},indent=2));return 0
        if args.command=='demo':
            result=demo(args.output,args.depth,args.integer_blocks)
            print(json.dumps({'project':'Compute Market','summary':str(args.output/'summary.json'),
                'case_sha256':result['case_sha256']},indent=2));return 0
        if args.command=='research':
            from .research import run_study
            options=vars(args).copy();options.pop('command')
            run_study(**options)
            print(json.dumps({'project':'Compute Market','summary':str(args.output/'summary.json')},indent=2));return 0
        if args.command=='replay':
            from .hedge_paths import FinancialTape
            from .risk_execution import ConditionalKernel
            from .execution import ExecutionMandate
            from .parallel import simulate_parallel,subset_tape
            tape=FinancialTape.load(args.folder/'holdout_tape.npz')
            count=tape.physical.config.paths if args.paths is None else args.paths
            if not 0<count<=tape.physical.config.paths:raise ValueError('replay path count must be within the saved tape')
            if count<tape.physical.config.paths:tape=subset_tape(tape,np.arange(count))
            kernel=ConditionalKernel.from_dict(json.loads((args.folder/'kernel.json').read_text()))
            mandate=ExecutionMandate(**json.loads((args.folder/'execution_mandate.json').read_text()))
            run=simulate_parallel(tape,kernel,args.policy,workers=args.workers,mandate=mandate)
            with np.load(args.folder/f'holdout_{args.policy}.npz',allow_pickle=False) as expected:
                checks={key:bool(np.array_equal(expected[key][:count] if expected[key].ndim else expected[key],getattr(run,key))) for key in expected.files}
            print(json.dumps({'paths':count,'policy':args.policy,'matches':checks,'all_match':all(checks.values())},indent=2))
            return 0 if all(checks.values()) else 2
        case=ValuationCase.load(args.case)
        if args.command=='validate':
            result={'status':'valid','case_sha256':case.fingerprint,'case_name':case.name,
                'nodes':case.portfolio.tree.size,'leaves':len(case.portfolio.tree.leaves),
                'offered_contract':case.offered is not None,'existing_book_explicit':True}
        elif args.command=='solve':result=case.solve(args.mode).to_dict()
        elif args.command=='quote':result=case.quote(mode=args.mode,payment=args.payment,price_tolerance=args.tolerance)
        else:
            result=case.verify_result(json.loads(args.result.read_text()))
            print(json.dumps(result,indent=2));return 0
        if args.output:write_json(args.output,result)
        else:print(json.dumps(result,indent=2,allow_nan=False))
        return 2 if result.get('status')=='infeasible' else 0
    except (ValueError,TypeError,KeyError,OSError,OptimizationError,RuntimeError) as exc:
        print(f'error: {exc}',file=sys.stderr);return 1

if __name__=='__main__':raise SystemExit(main())
