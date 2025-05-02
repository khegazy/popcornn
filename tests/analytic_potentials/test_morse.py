import os
from popcornn.tools.tests import popcornn_run_test

def test_morse():
    popcornn_run_test(
        name='morse',
        config_path=os.path.join('configs', 'morse.yaml'),
        benchmark_path=os.path.join('analytic_potentials', 'benchmarks')
    )