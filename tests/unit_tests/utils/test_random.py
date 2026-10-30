from simtools.utils import random


def test_seeds():
    fixed_seed = 123

    test_seed_1 = random.seeds(n_seeds=1, fixed_seed=fixed_seed)
    test_seed_2 = random.seeds(n_seeds=1, fixed_seed=fixed_seed)
    test_seed_3 = random.seeds(n_seeds=1, fixed_seed=None)
    assert isinstance(test_seed_1, int)
    assert test_seed_1 == test_seed_2
    assert test_seed_1 != test_seed_3

    test_seeds_list_1 = random.seeds(n_seeds=5, fixed_seed=fixed_seed)
    test_seeds_list_2 = random.seeds(n_seeds=5, fixed_seed=fixed_seed)
    test_seeds_list_3 = random.seeds(n_seeds=5, fixed_seed=None)
    assert isinstance(test_seeds_list_1, list)
    assert len(test_seeds_list_1) == 5
    assert test_seeds_list_1 == test_seeds_list_2

    assert test_seeds_list_1 != test_seeds_list_3


def test_instance_seeds():
    assert random.instance_seeds(42, 1) == [42]
    seeds = random.instance_seeds(42, 3, max_seed=100)
    assert seeds == random.seeds(n_seeds=3, max_seed=100, fixed_seed=42)
    assert all(1 <= seed < 100 for seed in seeds)
