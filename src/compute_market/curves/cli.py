"""Commands for the shared curve state and its numerical valuation methods."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
from .portfolio import CurveCase,example_case,EconomyTape
from .study import write,plain


def main(argv=None):
    p=argparse.ArgumentParser(description='Compute Market: stochastic curves, portfolio valuation and control')
    s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('example',help='write a complete synthetic curve case');a.add_argument('--output',type=Path,default=Path('examples/curve_case.json'))
    a=s.add_parser('validate',help='validate the complete curve case');a.add_argument('case',type=Path)
    a=s.add_parser('surface',help='reconstruct curves, discount factors and terminal forwards');a.add_argument('case',type=Path);a.add_argument('--output',type=Path,default=Path('results/surface.json'))
    a=s.add_parser('quote',help='price a contract using a curve-generated finite control program');a.add_argument('case',type=Path);a.add_argument('--branches',type=int,default=16);a.add_argument('--periods',type=int,default=2);a.add_argument('--seed',type=int,default=611);a.add_argument('--output',type=Path,default=Path('results/quote.json'))
    a=s.add_parser('study',help='run training, holdout, stresses and pricing experiments');a.add_argument('--output',type=Path,default=Path('results/study'));a.add_argument('--quick',action='store_true');a.add_argument('--workers',type=int,default=4)
    for name,default in [('paths',1024),('training-paths',4096),('stress-paths',512),('atoms',256)]:a.add_argument('--'+name,type=int,default=default)
    a=s.add_parser('calibrate',help='fit declared market and transition observations');a.add_argument('inputs',type=Path);a.add_argument('--output',type=Path,default=Path('results/calibration.json'))
    a=s.add_parser('analyze',help='run bucket risk, fleet options and synthetic calibration');a.add_argument('--output',type=Path,default=Path('results/study'));a.add_argument('--quick',action='store_true')
    a=s.add_parser('replay',help='replay a saved holdout from its exact factor and physical tape');a.add_argument('folder',type=Path);a.add_argument('--policy',choices=['unhedged','static','adaptive'],default='adaptive');a.add_argument('--output',type=Path)
    args=p.parse_args(argv)
    try:
        if args.command=='example':
            from .portfolio import project_case
            c=project_case();c.save(args.output);out={'case':str(args.output),'case_sha256':c.fingerprint}
        elif args.command=='study':
            from .study import run_study
            opts=vars(args).copy();opts.pop('command');run_study(**opts);out={'study':str(args.output/'summary.json')}
        elif args.command=='calibrate':
            from .calibration import fit_input_file
            args.output.parent.mkdir(parents=True,exist_ok=True)
            fit=fit_input_file(args.inputs,args.output);out={'fit':str(args.output),'success':fit.report['success'],'data_rank':fit.report['data_rank'],'parameters':fit.report['parameter_count']}
        elif args.command=='analyze':
            from .system_study import run_system_study
            run_system_study(args.output,args.quick);out={'analysis':str(args.output/'system_summary.json')}
        elif args.command=='replay':
            from .model import CurveModel
            from .continuation import ContinuationSurface
            from .control import run_controller,ConditionalInnovations
            root=args.folder;c=CurveCase.load(root/'case.json');t=EconomyTape.load(root/'holdout_tape.npz');v=ContinuationSurface.load(root/'continuation.npz')
            pm=CurveModel.from_dict(json.loads((root/'policy_model.json').read_text()));protocol=json.loads((root/'protocol.json').read_text())
            r=run_controller(c,t,v,policy=args.policy,scenarios=protocol['conditional_atoms'],seed=protocol['conditional_seed'],policy_model=pm,
                alpha=protocol['alpha_conditional'],risk_weight=protocol['risk_weight'],carry=protocol['carry'],
                innovations=ConditionalInnovations.load(root/'conditional_innovations.npz'))
            with np.load(root/f'holdout_{args.policy}.npz',allow_pickle=False) as z:
                checks={k:bool(np.array_equal(z[k],getattr(r,k))) for k in z.files if k!='metadata'}
            out={'policy':args.policy,'paths':len(r.terminal_wealth),'matches':checks,'all_match':all(checks.values()),'audit':r.audit}
            if args.output:write(args.output,out)
            print(json.dumps(plain(out),indent=2));return 0 if out['all_match'] else 2
        else:
            c=CurveCase.load(args.case)
            if args.command=='validate':out={'status':'valid','case_sha256':c.fingerprint,'model_sha256':c.model.fingerprint,'factors':c.model.dimension,'curves':list(c.model.curves),'instruments':len(c.instruments),'existing_book_explicit':True}
            elif args.command=='surface':
                x=np.zeros(c.model.dimension);dates=np.linspace(0,3,25)
                out={'case_sha256':c.fingerprint,'calendar_dates':dates,'futures':{k:[c.model.value(k,0,x,t) for t in dates] for k in c.model.curves},'discounts':[c.model.discount(0,x,t) for t in dates],
                     'terminal_forwards':{k:[c.model.forward(k,0,x,t) for t in dates] for k,v in c.model.curves.items() if v.kind!='funding'}}
                write(args.output,out);out={'surface':str(args.output),'case_sha256':c.fingerprint}
            else:
                from .adapter import finite_case
                case=finite_case(c,args.branches,args.periods,args.seed);q=case.quote(price_tolerance=.05)
                case.save(args.output.with_name(args.output.stem+'_finite_case.json'));write(args.output,q)
                out={'quote':str(args.output),'reservation':q['reservation'],'curve_case_sha256':c.fingerprint,'finite_case_sha256':case.fingerprint,'pricing_certificate':q['pricing_certificate']}
        print(json.dumps(plain(out),indent=2));return 0
    except (ValueError,TypeError,KeyError,OSError,RuntimeError) as e:
        print(f'error: {e}',file=sys.stderr);return 1
