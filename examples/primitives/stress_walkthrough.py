"""Compare a synthetic hedge under baseline and basis-and-funding stress."""
from compute_market.simulation import SimulationConfig,HedgePolicy,generate_tape,simulate
from compute_market.stress import StressScenario,apply_stress
from compute_market.statistics import simulation_summary,paired_comparison
import json

if __name__=='__main__':
    tape=generate_tape(SimulationConfig(paths=1000,steps=24,seed=17))
    policy=HedgePolicy('adaptive',.5,.7)
    base=simulate(tape,policy)
    stress=StressScenario('basis_funding','Local rental weakness and lender haircut.',
                         local_multiplier=(.6,.6,.8,.8),eligible_haircut=.2,draw_freeze=True)
    shocked=simulate(apply_stress(tape,stress),policy)
    print(json.dumps({'base':simulation_summary(base),'stress':simulation_summary(shocked),
                      'paired_stress_minus_base':paired_comparison(base.terminal_wealth,shocked.terminal_wealth)},indent=2))
