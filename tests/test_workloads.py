import pytest
from compute_market.workloads import Cluster,Workload,Outage,simulate_workloads


def test_exact_work_completion():
    r=simulate_workloads([Cluster('a',8,2,3)],[Workload('job',0,10,32,4,('a',))])
    assert r['completed']==1
    assert r['jobs'][0]['completion']==pytest.approx(4)
    assert r['total_cost']==pytest.approx(48)

def test_fragmented_capacity_cannot_fake_one_cluster():
    r=simulate_workloads([Cluster('a',4,1,1),Cluster('b',4,1,1)],
                        [Workload('job',0,2,4,6,('a','b'))])
    assert r['completed']==0 and r['deadline_failures']==1

def test_checkpoint_loss_retains_wasted_gpu_cost():
    r=simulate_workloads([Cluster('a',4,1,1)],
        [Workload('job',0,10,8,4,('a',),checkpoint_hours=1)], [Outage('a',.5,1.)])
    row=r['jobs'][0]
    assert row['lost_work']==pytest.approx(2)
    assert row['completion']==pytest.approx(3)
    assert row['gpu_hours']==pytest.approx(10)

def test_firm_interruption_flag():
    r=simulate_workloads([Cluster('a',4,1,1)],
        [Workload('job',0,10,8,4,('a',),interruption_allowed=False)], [Outage('a',.5,1.)])
    assert r['jobs'][0]['sla_breach']

def test_admission_does_not_see_future_outage():
    clusters=[Cluster('a',4,1,1),Cluster('b',4,1,2)]
    jobs=[Workload('j',0,10,16,4,('a','b'))]
    a=simulate_workloads(clusters,jobs)
    b=simulate_workloads(clusters,jobs,[Outage('a',2,3)])
    assert a['history'][0]['allocated_gpus']==b['history'][0]['allocated_gpus']

def test_no_overallocation_with_arrivals():
    r=simulate_workloads([Cluster('a',4,1,1)],
                        [Workload(str(i),i*.1,10,4,2,('a',)) for i in range(5)])
    assert r['max_gpu_overallocation']==0 and r['completed']==5

def test_same_time_completion_precedes_deadline():
    r=simulate_workloads([Cluster('a',1,1,1)],[Workload('j',0,1,1,1,('a',))])
    assert r['completed']==1

def test_missing_cluster_rejected():
    with pytest.raises(ValueError):
        simulate_workloads([Cluster('a',1,1,1)],[Workload('j',0,1,1,1,('b',))])
