import os
from popcornn.tools.tests import popcornn_run_test

def test_geodesic_initialize():
    popcornn_run_test(
        name='geodesic',
        config_path=os.path.join('configs', 'initialize_geodesic.yaml'),
        benchmark_path=os.path.join('initialize', 'benchmarks')
    )