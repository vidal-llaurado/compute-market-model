import json
from compute_market.experiments import run_suite


def test_complete_research_workflow(tmp_path):
    result=run_suite(tmp_path,paths=4,stress_paths=4,train_paths=4,steps=3,seed=21,bootstrap=2,quick=True)
    assert result['project']=='Compute Market' and result['synthetic']
    for name in ['summary.json','training.json','holdout.json','stresses.json','reverse_stress.json',
                 'optimizer_stresses.json','liquidity_surface.json','precision.json',
                 'parameter_sensitivity.json','importance_sampling.json','workload_stress.json']:
        data=json.loads((tmp_path/name).read_text())
        assert data
    assert (tmp_path/'holdout_paths.npz').exists()
