from typing import NamedTuple

from bong.types import Array


class AgentState(NamedTuple):
    """Belief state of an agent.

    mean: Mean of the belief state.
    cov: Covariance of the belief state.
    """

    mean: Array
    cov: Array


class DLRAgentState(NamedTuple):
    """Belief state of a DLR agent.

    mean: Mean of the belief state.
    prec_diag: Diagonal term (Upsilon) of DLR approximation of precision.
    prec_lr: Low-rank term (W) of DLR approximation of precision.
    """

    mean: Array
    prec_diag: Array
    prec_lr: Array
