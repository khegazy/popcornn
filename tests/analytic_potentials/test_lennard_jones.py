import os
from popcornn.tools.tests import popcornn_run_test

def test_lennard_jones():
    popcornn_run_test(
        name='lennard_jones',
        config_path=os.path.join('configs', 'lennard_jones.yaml'),
        benchmark_path=os.path.join('analytic_potentials', 'benchmarks')
    )