import os
from popcornn.tools.tests import popcornn_run_test

def test_harmonic():
    popcornn_run_test(
        'harmonic',
        os.path.join('configs', 'harmonic.yaml'),
        os.path.join('analytic_potentials', 'benchmarks')
    )