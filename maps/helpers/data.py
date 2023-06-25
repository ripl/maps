#!/usr/bin/env python3
from __future__ import annotations

import itertools
from copy import deepcopy
from typing import Any, Callable, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate


# Copied from pfnet/pfrl
def _to_recursive(batched: Any, device: torch.device) -> Any:
    if isinstance(batched, torch.Tensor):
        return batched.to(device)
    elif isinstance(batched, list):
        return [x.to(device) for x in batched]
    elif isinstance(batched, tuple):
        return tuple(x.to(device) for x in batched)
    elif isinstance(batched, dict):
        for val in batched.values():
            assert not isinstance(val, dict)
        return {key: x.to(device) for key, x in batched.items()}
    else:
        raise TypeError("Unsupported type of data")


# Copied from pfnet/pfrl
def batch_states(
    states: Sequence[Any], device: Optional[torch.device] = None, phi: Callable[[Any], Any] = lambda x: x
) -> Any:
    """The default method for making batch of observations.

    Args:
        states (list): list of observations from an environment.
        device (module): CPU or GPU the data should be placed on
        phi (callable): Feature extractor applied to observations

    Return:
        the object which will be given as input to the model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    features = [phi(s) for s in states]
    # return concat_examples(features, device=device)
    collated_features = default_collate(features)
    if isinstance(features[0], tuple):
        collated_features = tuple(collated_features)
    return _to_recursive(collated_features, device)


def yield_batch_infinitely(transitions: Sequence[dict], batch_size: int):
    import torch
    while True:
        indices = np.random.randint(low=0, high=len(transitions)-1, size=(batch_size,))
        sampled = [transitions[idx] for idx in indices]
        yield batch_states(sampled)


def flatten(array: Sequence[Iterable]):
    return list(itertools.chain.from_iterable(array))


def limit_num_transitions(episodes: List[List[dict]], max_transitions: int):
    """
    Truncate the episodes so that the number of total transitions is capped by `max_transitions`.

    Args:
        - episodes: list of episodes
    """
    assert isinstance(episodes, list) and isinstance(episodes[0], list), isinstance(episodes[0][0], dict)

    num_trans = 0
    cutoff_ep_idx, num_overrun = None, 0
    for idx, episode in enumerate(reversed(episodes)):
        # Keep track of the number of total transitions so far
        num_trans += len(episode)
        if num_trans >= max_transitions:
            # Compute number of overrun steps and which episode it reached `max_transitions`
            num_overrun = num_trans - max_transitions
            cutoff_ep_idx = idx
            break

    # Return as is if there's no need to truncate the episodes
    if cutoff_ep_idx is None or num_overrun is None:
        return episodes

    new_episodes = episodes[-cutoff_ep_idx-1:]  # Remove remaining old episodes
    new_episodes2 = deepcopy(new_episodes)
    new_episodes2[0] = new_episodes[0][num_overrun:]  # Remove old trajectories

    _num_transitions = len(flatten(new_episodes2))
    assert _num_transitions <= max_transitions
    return new_episodes2


def to_torch(x, device: Optional[torch.device] = None):
    import torch
    import numpy as np

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(x, (int, float)):
        return torch.as_tensor(x, device=device)
    elif isinstance(x, list):
        return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device, dtype=torch.float32)
    elif isinstance(x, np.ndarray):
        return torch.as_tensor(
            x,
            device=device,
            dtype=(torch.float32 if x.dtype in [np.float64, torch.float64] else None),
        )
    else:
        raise TypeError(f'Unsupported type: {type(x)}')
