"""Random numbers utilities."""

import secrets

import numpy as np


def seeds(n_seeds=1, min_seed=1, max_seed=2_000_000_000, fixed_seed=None):
    """
    Generate independent random seeds.

    Parameters
    ----------
    n_seeds : int
        Number of seeds to generate.
    min_seed : int
        Lower limit for the seed (inclusive).
    max_seed : int
        Upper limit for the seed (exclusive).
    fixed_seed : int or None
        If provided, use this fixed seed.

    Returns
    -------
    int or list of int:
        A single seed if n_seeds is 1, otherwise a list of seeds.
    """
    entropy = fixed_seed if fixed_seed is not None else secrets.randbits(128)
    ss = np.random.SeedSequence(entropy)
    rng = np.random.default_rng(ss)

    seed_list = rng.integers(low=min_seed, high=max_seed, size=n_seeds)

    if n_seeds == 1:
        return int(seed_list[0])
    return [int(x) for x in seed_list]
