import os
from popcornn.tools.tests import popcornn_run_test

def test_wolfe_schlegel():
    popcornn_run_test(
        name='wolfe_schlegel',
        config_path=os.path.join('configs', 'wolfe_schlegel.yaml'),
        benchmark_path=os.path.join('analytic_potentials', 'benchmarks')
    )