import os
from popcornn.tools.tests import popcornn_run_test

def test_repel():
    popcornn_run_test(
        name='repel',
        config_path=os.path.join('configs', 'repel.yaml'),
        benchmark_path=os.path.join('analytic_potentials', 'benchmarks')
    )