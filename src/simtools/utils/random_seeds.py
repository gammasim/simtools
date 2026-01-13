"""Random seeds utilities."""

import os
import secrets
import time

import numpy as np


def seeds(n_seeds=1, max_seed=2_000_000_000, fixed_seed=None):
    """
    Generate independent random seeds.

    Parameters
    ----------
    n_seeds : int
        Number of seeds to generate.
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

    seed_list = rng.integers(low=0, high=max_seed, size=n_seeds)

    if n_seeds == 1:
        return int(seed_list[0])
    return seed_list.tolist()


def seed_corsika_autoinputs_style(max_seed=2_000_000_000):
    """
    Generate random seeds as in corsika_autoinputs.

    Generates a high-entropy random seed based on system time,
    process ID, and /dev/urandom as the 'corsika_autoinputs' tool does.

    Parameters
    ----------
    max_seed (int):
        Upper limit for the seed (exclusive). Defaults to 2,000,000,000.

    Returns
    -------
    int:
        A random seed value between 0 and max_seed.
    """
    now = time.time()
    seconds = int(now)
    microseconds = int((now - seconds) * 1_000_000)
    pid = os.getpid()

    rnd_seed = (seconds % 147483647) + (2000 * microseconds) + (pid * 12345)

    try:
        random_bytes = os.urandom(4)
        r = int.from_bytes(random_bytes, byteorder="big")
        rnd_seed = (rnd_seed ^ r) % max_seed
    except (OSError, NotImplementedError):
        rnd_seed = rnd_seed % max_seed

    return rnd_seed
