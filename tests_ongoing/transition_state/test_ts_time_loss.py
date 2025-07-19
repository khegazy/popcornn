import os
from popcornn.tools.tests import popcornn_run_test

def test_ts_time_loss():
    benchmark_path = os.path.join(
        "transition_state", "benchmarks"
    )
    for potential in ['wolfe_schlegel', 'morse']:
        # Test high level functionality
        name = f"ts_time_loss_{potential}"
        config_path = os.path.join(
            "configs", f"{name}.yaml"
        )
        mep, _, _ = popcornn_run_test(name, config_path, benchmark_path)