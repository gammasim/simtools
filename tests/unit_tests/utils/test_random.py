from simtools.utils import random_seeds


def test_seeds():
    fixed_seed = 123

    test_seed_1 = random_seeds.seeds(n_seeds=1, fixed_seed=fixed_seed)
    test_seed_2 = random_seeds.seeds(n_seeds=1, fixed_seed=fixed_seed)
    test_seed_3 = random_seeds.seeds(n_seeds=1, fixed_seed=None)
    assert isinstance(test_seed_1, int)
    assert test_seed_1 == test_seed_2
    assert test_seed_1 != test_seed_3

    test_seeds_list_1 = random_seeds.seeds(n_seeds=5, fixed_seed=fixed_seed)
    test_seeds_list_2 = random_seeds.seeds(n_seeds=5, fixed_seed=fixed_seed)
    test_seeds_list_3 = random_seeds.seeds(n_seeds=5, fixed_seed=None)
    assert isinstance(test_seeds_list_1, list)
    assert len(test_seeds_list_1) == 5
    assert test_seeds_list_1 == test_seeds_list_2

    assert test_seeds_list_1 != test_seeds_list_3
