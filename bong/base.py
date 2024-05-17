from typing import Callable, NamedTuple


State = NamedTuple


class RebayesAlgorithm(NamedTuple):
    """Recursive Bayesian estimation algorithm.
    
    init: Initialize the algorithm.
    predict: Predict the next state.
    update: Update the belief state.
    sample: Sample from the belief state.
    scan: Run the algorithm over a sequence of observations.
    """
    init: Callable
    predict: Callable
    update: Callable
    sample: Callable
    name: str = 'short-name'
    full_name: str = 'full-name'